from PIL import Image
import os
import re 

from absl import app
from absl import flags

flags.DEFINE_string('pic_dir', "~/tmp/dump/nts_eval_img/episodes/1/1", 'task config.')
flags.DEFINE_string('output_file', '~/tmp/dump/nav.gif', 'output file')
flags.DEFINE_integer('frame_duration', 50, 'frame duration')
flags.DEFINE_string('pic_suffix', '.png', 'pic suffix')
flags.DEFINE_list('max_size', [100,100], 'max size')
flags.DEFINE_boolean('optimize', True, 'whether to optimize')

FLAGS = flags.FLAGS


def sorting_key(filename):
    match = re.search(r"(\d+)-(\d+)-Vis-(\d+)\.png$", filename)
    if match:
        return int(match.group(3))
    return 0  # Default if the pattern is not found


def resize_image(image, max_size):
    return image.resize(max_size, Image.LANCZOS)


def generate_gif_from_pics(
        pic_dir: str,
        output_file: str,
        frame_duration: int = 500,
        pic_suffix: str = '.png',
        max_size: tuple = (100, 100),
        sort_fn: callable = sorting_key,
        optimize: bool = True,
    ):
    # Sort the file names
    pic_dir = os.path.expanduser(pic_dir)
    file_names = sorted([os.path.join(pic_dir, f) for f in os.listdir(pic_dir) if f.endswith(pic_suffix)], key=sort_fn)

    # Create a list to hold the images
    images = []

    # Load each file, resize if necessary, and append to images list
    for file_name in file_names:
        with Image.open(file_name) as img:
            img_resized = resize_image(img, tuple(map(int, max_size)))
            images.append(img_resized.copy())  # Copy to ensure the file is not left open

    # Create and save the GIF
    gif_path = os.path.expanduser(output_file)
    images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=optimize, duration=frame_duration, loop=0)


def main(_):
    generate_gif_from_pics(FLAGS.pic_dir, FLAGS.output_file, FLAGS.frame_duration, FLAGS.pic_suffix, FLAGS.max_size, optimize=FLAGS.optimize)


if __name__ == '__main__':
    app.run(main)
