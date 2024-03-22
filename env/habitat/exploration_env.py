from typing import NamedTuple, List
import math
import os
import pickle
import sys
import functools

import gym
import matplotlib
import numpy as np
import quaternion
import skimage.morphology
import torch
from torch import nn 
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

import dm_env
from acme import specs


if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import env.habitat.habitat_api.habitat as habitat
from habitat import logger

from env.utils.map_builder import MapBuilder
from env.utils.fmm_planner import FMMPlanner

from env.habitat.utils.noisy_actions import CustomActionSpaceConfiguration
import env.habitat.utils.pose as pu
import env.habitat.utils.visualizations as vu
from env.habitat.utils.supervision import HabitatMaps
from mcts import types as mp_types


def get_grid(pose, grid_size, device):
    """ Neural-SLAM get_grid method
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size), align_corners=True)
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size), align_corners=True)

    return rot_grid, trans_grid


def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*1000.
    return depth


def action2goal(a: int, local_w: int, local_h: int, action_width: int , action_height: int):
    global_goals = [
        int((a % action_width + 0.5) / action_width * local_w),
        int((a // action_width + 0.5) / action_height * local_h)
        ] 
    return global_goals


def get_local_map_boundaries(agent_loc, local_sizes, full_sizes, args):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]


class ChannelPool(nn.MaxPool1d):
    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = x.contiguous()
        pooled = F.max_pool1d(x, c, 1)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)
    

class MapAggregater(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._args = args
        self.device = 'cpu'
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.vision_range = args.vision_range
        self.resolution = args.map_resolution
        self.agent_view = torch.zeros(1, 2, self.map_size_cm // self.resolution, self.map_size_cm // self.resolution).float().to(self.device)
        self.pool = ChannelPool(1)
    
    def forward(self, fp_proj, fp_exp, poses, pose_pred, local_map, local_explored, current_poses):
        with torch.no_grad():
            agent_view = self.agent_view.detach_()
            agent_view.fill_(0.)

            x1 = self.map_size_cm // (self.resolution * 2) \
                    - self.vision_range // 2
            x2 = x1 + self.vision_range
            y1 = self.map_size_cm // (self.resolution * 2)
            y2 = y1 + self.vision_range
            agent_view[:, :1, y1:y2, x1:x2] = torch.from_numpy(fp_proj)
            agent_view[:, 1:, y1:y2, x1:x2] = torch.from_numpy(fp_exp)

            corrected_pose = np.expand_dims(poses + pose_pred, axis=0)

            def get_new_pose_batch(pose, rel_pose_change):
                pose[:, 1] += rel_pose_change[:, 0] * \
                                torch.sin(pose[:, 2] / 57.29577951308232) \
                                + rel_pose_change[:, 1] * \
                                torch.cos(pose[:, 2] / 57.29577951308232)
                pose[:, 0] += rel_pose_change[:, 0] * \
                                torch.cos(pose[:, 2] / 57.29577951308232) \
                                - rel_pose_change[:, 1] * \
                                torch.sin(pose[:, 2] / 57.29577951308232)
                pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

                pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
                pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

                return pose

            current_poses = get_new_pose_batch(torch.from_numpy(np.expand_dims(current_poses, axis=0)),
                                                torch.from_numpy(corrected_pose))
            st_pose = current_poses.clone().detach()

            st_pose[:, :2] = - (st_pose[:, :2] * 100.0 / self.resolution
                                - self.map_size_cm \
                                // (self.resolution * 2)) \
                                / (self.map_size_cm // (self.resolution * 2))
            st_pose[:, 2] = 90. - (st_pose[:, 2])

            rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                            self.device)

            rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
            translated = F.grid_sample(rotated, trans_mat, align_corners=True)

            maps, explored = torch.from_numpy(np.expand_dims(local_map, axis=0)), torch.from_numpy(np.expand_dims(local_explored, axis=0))
            maps2 = torch.cat((maps.unsqueeze(1),
                                translated[:, :1, :, :]), 1)
            explored2 = torch.cat((explored.unsqueeze(1),
                                    translated[:, 1:, :, :]), 1)
            
            map_pred = (self.pool(maps2).squeeze(1).squeeze(0)).numpy()
            exp_pred = (self.pool(explored2).squeeze(1).squeeze(0)).numpy()
        return map_pred, exp_pred


class Exploration_Env(habitat.RLEnv):

    def __init__(self, args, rank, config_env, config_baseline, dataset):
        if args.visualize:
            plt.ion()
        if args.print_images or args.visualize:
            self.figure, self.ax = plt.subplots(1,2, figsize=(6*16/9, 6),
                                                facecolor="whitesmoke",
                                                num="Thread {}".format(rank))

        self.args = args

        map_size = self.args.map_size_cm // self.args.map_resolution
        self._full_w, self._full_h = map_size, map_size
        self._local_w, self._local_h = int(self._full_w / self.args.global_downscaling), int(self._full_h / self.args.global_downscaling)
        self._map_agg = MapAggregater(args)

        self.num_actions = 3
        self.dt = 10

        self.rank = rank

        self.sensor_noise_fwd = \
                pickle.load(open("noise_models/sensor_noise_fwd.pkl", 'rb'))
        self.sensor_noise_right = \
                pickle.load(open("noise_models/sensor_noise_right.pkl", 'rb'))
        self.sensor_noise_left = \
                pickle.load(open("noise_models/sensor_noise_left.pkl", 'rb'))
        
        if not habitat.SimulatorActions.has_action("NOISY_FORWARD"):
            habitat.SimulatorActions.extend_action_space("NOISY_FORWARD")
        if not habitat.SimulatorActions.has_action("NOISY_RIGHT"):
            habitat.SimulatorActions.extend_action_space("NOISY_RIGHT")
        if not habitat.SimulatorActions.has_action("NOISY_LEFT"):
            habitat.SimulatorActions.extend_action_space("NOISY_LEFT")

        config_env.defrost()
        config_env.SIMULATOR.ACTION_SPACE_CONFIG = \
                "CustomActionSpaceConfiguration"
        config_env.freeze()


        super().__init__(config_env, dataset)

        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.observation_space = gym.spaces.Box(0, 1,
                                                (3, args.frame_height,
                                                    args.frame_width),
                                                dtype='float32')

        self.mapper = self.build_mapper()

        self.episode_no = 0

        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.NEAREST)])
        self.scene_name = None
        self.maps_dict = {}

    def randomize_env(self):
        self._env._episode_iterator._shuffle_iterator()

    def save_trajectory_data(self):
        if "replica" in self.scene_name:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-3]+"/"
        else:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-1].split(".")[0]+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = folder+str(self.episode_no)+".txt"
        with open(filepath, "w+") as f:
            f.write(self.scene_name+"\n")
            for state in self.trajectory_states:
                f.write(str(state)+"\n")
            f.flush()

    def save_position(self):
        self.agent_state = self._env.sim.get_agent_state()
        self.trajectory_states.append([self.agent_state.position,
                                       self.agent_state.rotation])


    def reset(self):
        args = self.args
        self.episode_no += 1
        self.timestep = 0
        self._previous_action = None
        self.trajectory_states = []

        if args.randomize_env_every > 0:
            if np.mod(self.episode_no, args.randomize_env_every) == 0:
                self.randomize_env()

        # Get Ground Truth Map
        self.explorable_map = None
        while self.explorable_map is None:
            obs = super().reset()
            full_map_size = args.map_size_cm//args.map_resolution
            self.explorable_map = self._get_gt_map(full_map_size)
        self.prev_explored_area = 0.

        # Preprocess observations
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = (np.asarray(self.res(rgb)) / 255).astype('float32')
        state = rgb
        depth = _preprocess_depth(obs['depth'])

        # Initialize map and pose
        self.map_size_cm = args.map_size_cm
        self.mapper.reset_map(self.map_size_cm)
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.curr_loc_gt = self.curr_loc
        self.last_loc_gt = self.curr_loc_gt
        self.last_loc = self.curr_loc
        self.last_sim_location = self.get_sim_location()

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))

        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = self.mapper.update_map(depth, mapper_gt_pose)
        self._fp_proj, self._fp_explored = (fp_proj).astype('float32'), (fp_explored).astype('float32')

        # Initialize variables
        self.scene_name = self.habitat_env.sim.config.SCENE
        self.visited = np.zeros(self.map.shape)
        self.visited_vis = np.zeros(self.map.shape)
        self.visited_gt = np.zeros(self.map.shape)
        self.collison_map = np.zeros(self.map.shape)
        self.col_width = 1

        self._pose = np.asarray([0., 0., 0.], dtype='float32')
        self._pose_pred = np.asarray([0., 0., 0.], dtype='float32')
        # Set info
        self.info = {
            'time': self.timestep,
            'fp_proj': self._fp_proj,
            'fp_explored': self._fp_explored,
            'sensor_pose': self._pose,
            'pose_err': self._pose_pred,
            'exp_reward': 0.,
            'exp_ratio': 0.,
            'map': self.explorable_map,
        }
        self.save_position()

        return state, self.info

    def step(self, action):

        args = self.args
        self.timestep += 1

        # Action remapping
        if action == 2: # Forward
            action = 1
            noisy_action = habitat.SimulatorActions.NOISY_FORWARD
        elif action == 1: # Right
            action = 3
            noisy_action = habitat.SimulatorActions.NOISY_RIGHT
        elif action == 0: # Left
            action = 2
            noisy_action = habitat.SimulatorActions.NOISY_LEFT

        self.last_loc = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)
        self._previous_action = action

        if args.noisy_actions:
            obs, rew, done, info = super().step(noisy_action)
        else:
            obs, rew, done, info = super().step(action)

        # Preprocess observations
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = (np.asarray(self.res(rgb)) / 255).astype('float32')

        state = rgb

        depth = _preprocess_depth(obs['depth'])

        # Get base sensor and ground-truth pose
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change()
        dx_base, dy_base, do_base = self.get_base_pose_change(
                                        action, (dx_gt, dy_gt, do_gt))

        self.curr_loc = pu.get_new_pose(self.curr_loc,
                               (dx_base, dy_base, do_base))

        self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,
                               (dx_gt, dy_gt, do_gt))

        if not args.noisy_odometry:
            self.curr_loc = self.curr_loc_gt
            dx_base, dy_base, do_base = dx_gt, dy_gt, do_gt

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))


        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = self.mapper.update_map(depth, mapper_gt_pose)
        self._fp_proj, self._fp_explored = (fp_proj).astype('float32'), (fp_explored).astype('float32')

        # Update collision map
        if action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, t2 = self.curr_loc
            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                self.col_width = min(self.col_width, 9)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold: #Collision
                length = 2
                width = self.col_width
                buf = 3
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r*100/args.map_resolution), \
                               int(c*100/args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                    self.collison_map.shape)
                        self.collison_map[r,c] = 1

        self._pose = np.asarray([dx_base, dy_base, do_base], dtype='float32')
        self._pose_pred = np.asarray([dx_gt - dx_base, dy_gt - dy_base, do_gt - do_base], dtype='float32')
        # Set info
        self.info['time'] = self.timestep
        self.info['fp_proj'] = self._fp_proj
        self.info['fp_explored']= self._fp_explored
        self.info['sensor_pose'] = self._pose
        self.info['pose_err'] = self._pose_pred 

        area, ratio = self.get_global_reward()
        rew = self.info['exp_reward'] = area
        self.info['exp_ratio'] = ratio
        self.info['map'] = self.explorable_map
        self.save_position()

        if self.info['time'] >= args.max_episode_length:
            done = True
            if self.args.save_trajectory_data != "0":
                self.save_trajectory_data()
        else:
            done = False

        return state, rew, done, self.info

    def get_reward_range(self):
        # This function is not used, Habitat-RLEnv requires this function
        return (0., 1.0)

    def get_reward(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return 0.

    def get_global_reward(self):
        curr_explored = self.explored_map*self.explorable_map
        curr_explored_area = curr_explored.sum()

        reward_scale = self.explorable_map.sum()
        m_reward = (curr_explored_area - self.prev_explored_area)*1.
        m_ratio = m_reward/reward_scale
        m_reward = m_reward * 25./10000. # converting to m^2
        self.prev_explored_area = curr_explored_area

        m_reward *= 0.02 # Reward Scaling

        return m_reward, m_ratio

    def get_done(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return False

    def get_info(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        info = {}
        return info

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def get_spaces(self):
        return self.observation_space, self.action_space

    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] =  self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper


    def get_sim_location(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o


    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do


    def get_base_pose_change(self, action, gt_pose_change):
        dx_gt, dy_gt, do_gt = gt_pose_change
        if action == 1: ## Forward
            x_err, y_err, o_err = self.sensor_noise_fwd.sample()[0][0]
        elif action == 3: ## Right
            x_err, y_err, o_err = self.sensor_noise_right.sample()[0][0]
        elif action == 2: ## Left
            x_err, y_err, o_err = self.sensor_noise_left.sample()[0][0]
        else: ##Stop
            x_err, y_err, o_err = 0., 0., 0.

        x_err = x_err * self.args.noise_level
        y_err = y_err * self.args.noise_level
        o_err = o_err * self.args.noise_level
        return dx_gt + x_err, dy_gt + y_err, do_gt + np.deg2rad(o_err)


    def get_short_term_goal(self, planner_input: mp_types.PlannerInput):
        goal = planner_input.goal
        map_pred = planner_input.map_prediction if planner_input.map_prediction.size else self._fp_proj
        exp_pred = planner_input.explored_prediction if planner_input.explored_prediction.size else self._fp_explored
        pos_pred = planner_input.pos_prediction if planner_input.pos_prediction.size else self._pose_pred
        args = self.args
        # Get pose prediction and global policy planning window
        start_x, start_y, start_o = self.curr_loc_gt
        r, c = start_y, start_x
        agent_loc = [int(r * 100.0/args.map_resolution),
                      int(c * 100.0/args.map_resolution)]
        
        gx1, gx2, gy1, gy2 = get_local_map_boundaries(agent_loc, (self._local_w, self._local_h), (self._full_w, self._full_h), args)
        # Get Map prediction
        local_map, local_explored, cur_pose = self.map[gx1:gx2, gy1:gy2], self.explored_map[gx1:gx2, gy1:gy2], np.asarray(self.curr_loc_gt)
        map_pred, exp_pred = self._map_agg(map_pred, exp_pred, self._pose, pos_pred, local_map, local_explored, cur_pose)

        grid = np.rint(map_pred)
        explored = np.rint(exp_pred)

        goal = pu.threshold_poses(goal, grid.shape)
        
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        
        # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0/args.map_resolution - gx1),
                      int(c * 100.0/args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, grid.shape)

        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0/args.map_resolution - gx1),
                 int(c * 100.0/args.map_resolution - gy1)]
        start = pu.threshold_poses(start, grid.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1

        steps = args.num_local_steps
        for i in range(steps):
            x = int(last_start[0] + (start[0] - last_start[0]) * (i+1) / steps)
            y = int(last_start[1] + (start[1] - last_start[1]) * (i+1) / steps)
            self.visited_vis[gx1:gx2, gy1:gy2][x, y] = 1

        # Get last loc ground truth pose
        last_start_x, last_start_y = self.last_loc_gt[0], self.last_loc_gt[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0/args.map_resolution),
                      int(c * 100.0/args.map_resolution)]
        last_start = pu.threshold_poses(last_start, self.visited_gt.shape)

        # Get ground truth pose
        start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt
        r, c = start_y_gt, start_x_gt
        start_gt = [int(r * 100.0/args.map_resolution),
                    int(c * 100.0/args.map_resolution)]
        start_gt = pu.threshold_poses(start_gt, self.visited_gt.shape)
        #self.visited_gt[start_gt[0], start_gt[1]] = 1

        steps = args.num_local_steps
        for i in range(steps):
            x = int(last_start[0] + (start_gt[0] - last_start[0]) * (i+1) / steps)
            y = int(last_start[1] + (start_gt[1] - last_start[1]) * (i+1) / steps)
            self.visited_gt[x, y] = 1

        # Get intrinsic reward for global policy
        # Negative reward for exploring explored areas i.e.
        # for choosing explored cell as long-term goal
        self.extrinsic_rew = -pu.get_l2_distance(10, goal[0], 10, goal[1])
        self.intrinsic_rew = -exp_pred[goal[0], goal[1]]

        # Get short-term goal
        stg = self._get_stg(grid, explored, start, np.copy(goal), planning_window)

        # Find GT action
        gt_action = self._get_gt_action(1 - self.explorable_map, start,
                                        [int(stg[0]), int(stg[1])],
                                        planning_window, start_o)

        (stg_x, stg_y) = stg
        relative_dist = pu.get_l2_distance(stg_x, start[0], stg_y, start[1])
        relative_dist = relative_dist*5./100.
        angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                stg_y - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        def discretize(dist):
            dist_limits = [0.25, 3, 10]
            dist_bin_size = [0.05, 0.25, 1.]
            if dist < dist_limits[0]:
                ddist = int(dist/dist_bin_size[0])
            elif dist < dist_limits[1]:
                ddist = int((dist - dist_limits[0])/dist_bin_size[1]) + \
                    int(dist_limits[0]/dist_bin_size[0])
            elif dist < dist_limits[2]:
                ddist = int((dist - dist_limits[1])/dist_bin_size[2]) + \
                    int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1])
            else:
                ddist = int(dist_limits[0]/dist_bin_size[0]) + \
                    int((dist_limits[1] - dist_limits[0])/dist_bin_size[1]) + \
                    int((dist_limits[2] - dist_limits[1])/dist_bin_size[2])
            return ddist

        self.relative_angle = relative_angle

        if (args.visualize or args.print_images) and self.timestep % args.plot_every == 0:
            dump_dir = "{}/dump/{}/".format(args.dump_location,
                                                args.exp_name)
            ep_dir = '{}/episodes/{}/{}/'.format(
                            dump_dir, self.rank+1, self.episode_no)
            if not os.path.exists(ep_dir):
                os.makedirs(ep_dir)

            if args.vis_type == 1: # Visualize predicted map and pose
                vis_grid = vu.get_colored_map(np.rint(map_pred),
                                self.collison_map[gx1:gx2, gy1:gy2],
                                self.visited_vis[gx1:gx2, gy1:gy2],
                                self.visited_gt[gx1:gx2, gy1:gy2],
                                goal,
                                self.explored_map[gx1:gx2, gy1:gy2],
                                self.explorable_map[gx1:gx2, gy1:gy2],
                                self.map[gx1:gx2, gy1:gy2] *
                                    self.explored_map[gx1:gx2, gy1:gy2])
                vis_grid = np.flipud(vis_grid)
                vu.visualize(self.figure, self.ax, self.obs, vis_grid[:,:,::-1],
                            (start_x - gy1*args.map_resolution/100.0,
                             start_y - gx1*args.map_resolution/100.0,
                             start_o),
                            (start_x_gt - gy1*args.map_resolution/100.0,
                             start_y_gt - gx1*args.map_resolution/100.0,
                             start_o_gt),
                            dump_dir, self.rank, self.episode_no,
                            self.timestep, args.visualize,
                            args.print_images, args.vis_type)

            else: # Visualize ground-truth map and pose
                vis_grid = vu.get_colored_map(self.map,
                                self.collison_map,
                                self.visited_gt,
                                self.visited_gt,
                                goal,
                                # (goal[0]+gx1, goal[1]+gy1),
                                self.explored_map,
                                self.explorable_map,
                                self.map*self.explored_map)
                vis_grid = np.flipud(vis_grid)
                vu.visualize(self.figure, self.ax, self.obs, vis_grid[:,:,::-1],
                            (start_x_gt, start_y_gt, start_o_gt),
                            (start_x_gt, start_y_gt, start_o_gt),
                            dump_dir, self.rank, self.episode_no,
                            self.timestep, args.visualize,
                            args.print_images, vis_style=0)

        return gt_action

    def _get_gt_map(self, full_map_size):
        self.scene_name = self.habitat_env.sim.config.SCENE
        logger.error('Computing map for %s', self.scene_name)

        # Get map in habitat simulator coordinates
        self.map_obj = HabitatMaps(self.habitat_env)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            logger.error("Invalid map: {}/{}".format(
                            self.scene_name, self.episode_no))
            return None

        agent_y = self._env.sim.get_agent_state().position.tolist()[1]*100.
        sim_map = self.map_obj.get_map(agent_y, -50., 50.0)

        sim_map[sim_map > 0] = 1.

        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin/100.0
        x, y, o = self.get_sim_location()
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.

        map_size = sim_map.shape
        scale = 2.
        grid_size = int(scale*max(map_size))
        grid_map = np.zeros((grid_size, grid_size))

        grid_map[(grid_size - map_size[0])//2:
                 (grid_size - map_size[0])//2 + map_size[0],
                 (grid_size - map_size[1])//2:
                 (grid_size - map_size[1])//2 + map_size[1]] = sim_map

        if map_size[0] > map_size[1]:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale) \
                             * map_size[1] * 1. / map_size[0],
                    (y - range_y/2.) * 2. / (range_y * scale),
                    180.0 + np.rad2deg(o)
                ]])

        else:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale),
                    (y - range_y/2.) * 2. / (range_y * scale) \
                            * map_size[0] * 1. / map_size[1],
                    180.0 + np.rad2deg(o)
                ]])

        rot_mat, trans_mat = get_grid(st, (1, 1,
            grid_size, grid_size), torch.device("cpu"))

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        # Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. 
        # specify align_corners=True if the old behavior is desired.
        translated = F.grid_sample(grid_map, trans_mat, align_corners=True) 
        rotated = F.grid_sample(translated, rot_mat, align_corners=True)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > grid_size:
            episode_map[(full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size,
                        (full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size] = \
                                rotated[0,0]
        else:
            episode_map = rotated[0,0,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size]



        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.

        return episode_map


    def _get_stg(self, grid, explored, start, goal, planning_window):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(20., dist)
        x1 = max(1, int(x1 - buf))
        x2 = min(grid.shape[0]-1, int(x2 + buf))
        y1 = max(1, int(y1 - buf))
        y2 = min(grid.shape[1]-1, int(y2 + buf))

        rows = explored.sum(1)
        rows[rows>0] = 1
        ex1 = np.argmax(rows)
        ex2 = len(rows) - np.argmax(np.flip(rows))

        cols = explored.sum(0)
        cols[cols>0] = 1
        ey1 = np.argmax(cols)
        ey2 = len(cols) - np.argmax(np.flip(cols))

        ex1 = min(int(start[0]) - 2, ex1)
        ex2 = max(int(start[0]) + 2, ex2)
        ey1 = min(int(start[1]) - 2, ey1)
        ey2 = max(int(start[1]) + 2, ey2)

        x1 = max(x1, ex1)
        x2 = min(x2, ex2)
        y1 = max(y1, ey1)
        y2 = min(y2, ey2)

        traversible = skimage.morphology.binary_dilation(
                        grid[x1:x2, y1:y2],
                        self.selem) != True
        traversible[self.collison_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1

        traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                    int(start[1]-y1)-1:int(start[1]-y1)+2] = 1

        if goal[0]-2 > x1 and goal[0]+3 < x2\
            and goal[1]-2 > y1 and goal[1]+3 < y2:
            traversible[int(goal[0]-x1)-2:int(goal[0]-x1)+3,
                    int(goal[1]-y1)-2:int(goal[1]-y1)+3] = 1
        else:
            goal[0] = min(max(x1, goal[0]), x2)
            goal[1] = min(max(y1, goal[1]), y2)

        def add_boundary(mat):
            h, w = mat.shape
            new_mat = np.ones((h+2,w+2))
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        traversible = add_boundary(traversible)

        planner = FMMPlanner(traversible, 360//self.dt)

        reachable = planner.set_goal([goal[1]-y1+1, goal[0]-x1+1])

        stg_x, stg_y = start[0] - x1 + 1, start[1] - y1 + 1
        for i in range(self.args.short_goal_dist):
            stg_x, stg_y, replan = planner.get_short_term_goal([stg_x, stg_y])
        if replan:
            stg_x, stg_y = start[0], start[1]
        else:
            stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1

        return (stg_x, stg_y)


    def _get_gt_action(self, grid, start, goal, planning_window, start_o):

        [gx1, gx2, gy1, gy2] = planning_window

        x1 = min(start[0], goal[0])
        x2 = max(start[0], goal[0])
        y1 = min(start[1], goal[1])
        y2 = max(start[1], goal[1])
        dist = pu.get_l2_distance(goal[0], start[0], goal[1], start[1])
        buf = max(5., dist)
        x1 = max(0, int(x1 - buf))
        x2 = min(grid.shape[0], int(x2 + buf))
        y1 = max(0, int(y1 - buf))
        y2 = min(grid.shape[1], int(y2 + buf))

        path_found = False
        goal_r = 0
        while not path_found:
            traversible = skimage.morphology.binary_dilation(
                            grid[gx1:gx2, gy1:gy2][x1:x2, y1:y2],
                            self.selem) != True
            traversible[self.visited[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
            traversible[int(start[0]-x1)-1:int(start[0]-x1)+2,
                        int(start[1]-y1)-1:int(start[1]-y1)+2] = 1
            traversible[int(goal[0]-x1)-goal_r:int(goal[0]-x1)+goal_r+1,
                        int(goal[1]-y1)-goal_r:int(goal[1]-y1)+goal_r+1] = 1
            scale = 1
            planner = FMMPlanner(traversible, 360//self.dt, scale)

            reachable = planner.set_goal([goal[1]-y1, goal[0]-x1])

            stg_x_gt, stg_y_gt = start[0] - x1, start[1] - y1
            for i in range(1):
                stg_x_gt, stg_y_gt, replan = \
                        planner.get_short_term_goal([stg_x_gt, stg_y_gt])

            if replan and buf < 100.:
                buf = 2*buf
                x1 = max(0, int(x1 - buf))
                x2 = min(grid.shape[0], int(x2 + buf))
                y1 = max(0, int(y1 - buf))
                y2 = min(grid.shape[1], int(y2 + buf))
            elif replan and goal_r < 50:
                goal_r += 1
            else:
                path_found = True

        stg_x_gt, stg_y_gt = stg_x_gt + x1, stg_y_gt + y1
        angle_st_goal = math.degrees(math.atan2(stg_x_gt - start[0],
                                                stg_y_gt - start[1]))
        angle_agent = (start_o)%360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal)%360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > 15.:
            gt_action = 1
        elif relative_angle < -15.:
            gt_action = 0
        else:
            gt_action = 2

        return gt_action


class DMExploration(Exploration_Env):
    def __init__(self, args, rank, config_env, config_baseline, dataset):
        super().__init__(args, rank, config_env, config_baseline, dataset)
        self.num_actions = int(self.args.action_width * self.args.action_height)
        self._rgb_dim = np.prod((self.args.frame_height, self.args.frame_width, 3))
        self._fp_proj_dim = self._fp_explored_dim = np.prod((self.args.vision_range, self.args.vision_range))
        self._sensor_pose_dim = self._pose_err_dim = 3 
        self._exp_reward_dim = self._exp_ratio_dim = 1
        self._init_pos()
        self.observation_space = self.observation_spec()
        self._obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        self._reset_next_step = True

        self._action2goal = functools.partial(action2goal, local_h=self._local_h, local_w=self._local_w, action_height=self.args.action_height, action_width=self.args.action_width)

    def reset(self):
        self._reset_next_step = False
        state, self.info = super().reset()
        observation = self._observe(state, self.info)
        return dm_env.restart(observation)

    def step(self, action):
        if self._reset_next_step:
            return self.reset()
        # from map location to robot movement
        map_goal = self._action2goal(action)
        reward = 0.
        for step in range(self.args.num_local_steps): 
            a = self.get_short_term_goal(mp_types.PlannerInput(goal=map_goal))
            state, rew, done, self.info = super().step(a)
            reward += rew
            observation = self._observe(state, self.info)
            if done: 
                self._reset_next_step = True 
                return dm_env.termination(reward=reward, observation=observation)
        else:
            return dm_env.transition(reward=reward, observation=observation, discount=self.args.discount)
        
    def observation_spec(self):
        total_dim = self._rgb_dim + self._fp_proj_dim + self._fp_explored_dim  + self._sensor_pose_dim + self._pose_err_dim + self._exp_reward_dim + self._exp_ratio_dim
        observation_space = specs.BoundedArray((total_dim,), dtype='float32', minimum=0., maximum=1.)
        # {'state': specs.BoundedArray((self.args.frame_height, self.args.frame_width, 3), dtype='float32', minimum=0., maximum=1., name='normalized RGB'),
        #                      'info': {'time': specs.Array((),dtype=int),
        #                               'fp_proj': specs.BoundedArray((self.args.vision_range, self.args.vision_range), dtype='float32', minimum=0., maximum=1.),
        #                               'fp_explored': specs.BoundedArray((self.args.vision_range, self.args.vision_range), dtype='float32', minimum=0., maximum=1.),
        #                               'sensor_pose': specs.BoundedArray((3,), dtype='float32', minimum=0., maximum=1.),
        #                               'pose_err': specs.BoundedArray((3,), dtype='float32', minimum=0., maximum=1.),
        #                               'exp_reward': specs.Array((),dtype=float),
        #                               'exp_ratio': specs.Array((),dtype=float),
        #                               }}
        return observation_space
    
    def action_spec(self):
        action_space = specs.DiscreteArray(dtype='int32', num_values=self.num_actions, name='action')
        return action_space
        
    def reward_spec(self):
        return specs.Array((), dtype='float32')
    
    def discount_spec(self):
        return specs.Array((), dtype='float32')
    
    def _observe(self, state, info):
        self._obs[:self._rgb_dim] = state.flatten()
        self._obs[self._rgb_dim: self._p1] = info['fp_proj'].flatten()
        self._obs[self._p1: self._p2] = info['fp_explored'].flatten()
        self._obs[self._p2: self._p3] = info['sensor_pose'].flatten()
        self._obs[self._p3: self._p4] = info['pose_err'].flatten()
        self._obs[self._p4: self._p5] = info['exp_reward']
        self._obs[self._p5: self._p6] = info['exp_ratio']
        return self._obs 
    
    def _init_pos(self):
        self._p1 = self._rgb_dim + self._fp_proj_dim
        self._p2 = self._p1 + self._fp_explored_dim
        self._p3 = self._p2 + self._sensor_pose_dim
        self._p4 = self._p3 + self._pose_err_dim
        self._p5 = self._p4 + self._exp_reward_dim
        self._p6 = self._p5 + self._exp_ratio_dim


class DMExplorationExtras(Exploration_Env):
    def __init__(self, args, rank, config_env, config_baseline, dataset):
        super().__init__(args, rank, config_env, config_baseline, dataset)
        self.num_actions = int(self.args.action_width * self.args.action_height)
        self._rgb_dim = np.prod((self.args.frame_height, self.args.frame_width, 3))
        self._fp_proj_dim = self._fp_explored_dim = np.prod((self.args.vision_range, self.args.vision_range))
        self._sensor_pose_dim = self._pose_err_dim = 3 
        self._exp_reward_dim = self._exp_ratio_dim = 1
        self._init_pos()
        self.observation_space = self.observation_spec()
        self._obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        self._reset_next_step = True

        self._action2goal = functools.partial(action2goal, local_h=self._local_h, local_w=self._local_w, action_height=self.args.action_height, action_width=self.args.action_width)

    def reset(self):
        self._reset_next_step = False
        state, self.info = super().reset()
        observation = self._observe(state, self.info)
        return dm_env.restart(observation)

    def step(self, action: mp_types.ActionExtras):
        if self._reset_next_step:
            return self.reset()
        # from map location to robot movement
        map_goal = self._action2goal(action.action)
        a = self.get_short_term_goal(mp_types.PlannerInput(
            goal=map_goal, 
            map_prediction=action.map_prediction,
            explored_prediction=action.explored_prediction,
            pos_prediction=action.pos_prediction))
        state, reward, done, self.info = super().step(a)
        observation = self._observe(state, self.info)
        if done: 
            self._reset_next_step = True 
            return dm_env.termination(reward=reward, observation=observation)
        else:
            return dm_env.transition(reward=reward, observation=observation, discount=self.args.discount)
        
    def observation_spec(self):
        total_dim = self._rgb_dim + self._fp_proj_dim + self._fp_explored_dim  + self._sensor_pose_dim + self._pose_err_dim + self._exp_reward_dim + self._exp_ratio_dim
        observation_space = specs.BoundedArray((total_dim,), dtype='float32', minimum=0., maximum=1.)
        # {'state': specs.BoundedArray((self.args.frame_height, self.args.frame_width, 3), dtype='float32', minimum=0., maximum=1., name='normalized RGB'),
        #                      'info': {'time': specs.Array((),dtype=int),
        #                               'fp_proj': specs.BoundedArray((self.args.vision_range, self.args.vision_range), dtype='float32', minimum=0., maximum=1.),
        #                               'fp_explored': specs.BoundedArray((self.args.vision_range, self.args.vision_range), dtype='float32', minimum=0., maximum=1.),
        #                               'sensor_pose': specs.BoundedArray((3,), dtype='float32', minimum=0., maximum=1.),
        #                               'pose_err': specs.BoundedArray((3,), dtype='float32', minimum=0., maximum=1.),
        #                               'exp_reward': specs.Array((),dtype=float),
        #                               'exp_ratio': specs.Array((),dtype=float),
        #                               }}
        return observation_space
    
    def action_spec(self):
        action_space = mp_types.ActionExtras(
            action=specs.DiscreteArray(dtype='int32', num_values=self.num_actions, name='action'),
            map_prediction=specs.BoundedArray((self.args.vision_range, self.args.vision_range), minimum=0., maximum=1., dtype='float32'),
            explored_prediction=specs.BoundedArray((self.args.vision_range, self.args.vision_range), minimum=0., maximum=1., dtype='float32'),
            pos_prediction=specs.BoundedArray((3,), minimum=0., maximum=1., dtype='float32')
            )
        return action_space
        
    def reward_spec(self):
        return specs.Array((), dtype='float32')
    
    def discount_spec(self):
        return specs.Array((), dtype='float32')
    
    def _observe(self, state, info):
        self._obs[:self._rgb_dim] = state.flatten()
        self._obs[self._rgb_dim: self._p1] = info['fp_proj'].flatten()
        self._obs[self._p1: self._p2] = info['fp_explored'].flatten()
        self._obs[self._p2: self._p3] = info['sensor_pose'].flatten()
        self._obs[self._p3: self._p4] = info['pose_err'].flatten()
        self._obs[self._p4: self._p5] = info['exp_reward']
        self._obs[self._p5: self._p6] = info['exp_ratio']
        return self._obs 
    
    def _init_pos(self):
        self._p1 = self._rgb_dim + self._fp_proj_dim
        self._p2 = self._p1 + self._fp_explored_dim
        self._p3 = self._p2 + self._sensor_pose_dim
        self._p4 = self._p3 + self._pose_err_dim
        self._p5 = self._p4 + self._exp_reward_dim
        self._p6 = self._p5 + self._exp_ratio_dim

