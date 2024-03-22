import sys
import numpy as np 
from numpy import ma
import matplotlib
if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import skimage
import skfmm
import cv2 


def rotate_translate_map2d(pose, agent_view):
    x, y, t = pose 
    center = (agent_view.shape[1] / 2, agent_view.shape[0] / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, t, 1.)
    rotated_image = cv2.warpAffine(agent_view, rotation_matrix, (agent_view.shape[1], agent_view.shape[0]))
    translation_matrix = rotation_matrix
    translation_matrix[0, 2] += x  
    translation_matrix[1, 2] += y  
    translated_image = cv2.warpAffine(rotated_image, translation_matrix, (agent_view.shape[1], agent_view.shape[0]))
    return translated_image


def visualize(fig, ax, img, grid, pos, dump_dir, rank, ep_no, t,
              visualize, print_images, vis_style=1, map_resolution: float = 5.):
    for i in range(2):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    ax[0].imshow(img)
    ax[0].set_title("Observation", family='sans-serif', fontsize=20)

    if vis_style == 1:
        title = "Predicted Map and Pose"
    else:
        title = "Ground-Truth Map and Pose"

    ax[1].imshow(grid)
    ax[1].set_title(title, family='sans-serif', fontsize=20)

    # Draw predicted agent pose
    x, y, o = pos
    agent_size = 8
    x, y = x * 100.0 / map_resolution, grid.shape[1] - y * 100.0 / map_resolution

    dx = 0
    dy = 0
    fc = 'Red'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(x - 1 * dx, y - 1 * dy, dx * agent_size, dy * agent_size * 1.25,
                head_width=agent_size, head_length=agent_size * 1.25,
                length_includes_head=True, fc=fc, ec=fc, alpha=0.6)

    for _ in range(5):
        plt.tight_layout()

    if visualize:
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    if print_images:
        fn = '{}/episodes/{}/{}/{}-{}-Vis-{}.png'.format(
            dump_dir, (rank + 1), ep_no, rank, ep_no, t)
        plt.savefig(fn)


def fill_color(colored, mat, color):
    for i in range(3):
        colored[:, :, 2 - i] *= (1 - mat)
        colored[:, :, 2 - i] += (1 - color[i]) * mat
    return colored


def get_colored_map(mat, collision_map, visited, goal,):
    m, n = mat.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    green_palette = sns.light_palette("green")
    colored = fill_color(colored, mat, pal[2])

    current_palette = [(0.6, 0.6, 0.6)]

    red_palette = sns.light_palette("red")
    colored = fill_color(colored, visited, pal[4])

    colored = fill_color(colored, collision_map, pal[2])

    current_palette = sns.color_palette()

    selem = skimage.morphology.disk(4)
    goal_mat = np.zeros((m, n))
    goal_mat[goal[0], goal[1]] = 1
    goal_mat = 1 - skimage.morphology.binary_dilation(
        goal_mat, selem) != True

    colored = fill_color(colored, goal_mat, current_palette[0])

    current_palette = sns.color_palette("Paired")

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    return colored


class FMMPlanner():
    def __init__(self, traversible, num_rots, scale=1, step_size=5):
        self.scale = scale
        self.step_size = step_size
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale, traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.angle_value = [0, 2.0 * np.pi / num_rots, -2.0 * np.pi / num_rots, 0]
        self.du = int(self.step_size / (self.scale * 1.))
        self.num_rots = num_rots

    def set_goal(self, goal):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.)), \
                         int(goal[1] / (self.scale * 1.))
        traversible_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return dd_mask

    def get_short_term_goal(self, state):
        scale = self.scale * 1.
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = self._get_mask(dx, dy, scale, self.step_size)
        dist_mask = self._get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]

        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                 state[1]:state[1] + 2 * self.du + 1]

        assert subset.shape[0] == 2 * self.du + 1 and \
               subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2
        subset -= subset[self.du, self.du]
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        trav = np.pad(self.traversible, self.du,
                      'constant', constant_values=0)

        subset_trav = trav[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]
        traversible_ma = ma.masked_values(subset_trav * 1, 0)
        goal_x, goal_y = self.du, self.du
        traversible_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        subset_fmm_dist = dd

        subset_fmm_dist[subset_fmm_dist < 4] = 4.
        subset = subset / subset_fmm_dist
        subset[subset < -1.5] = 1
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False
        return (stg_x + state[0] - self.du) * scale + 0.5, \
               (stg_y + state[1] - self.du) * scale + 0.5, replan

    def _get_mask(self, sx, sy, scale, step_size):
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                        step_size ** 2:
                    mask[i, j] = 1
        return mask
    
    def _get_dist(self, sx, sy, scale, step_size):
        size = int(step_size // scale) * 2 + 1
        mask = np.zeros((size, size)) + 1e-10
        for i in range(size):
            for j in range(size):
                if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                        step_size ** 2:
                    mask[i, j] = max(5, (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                        ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
        return mask