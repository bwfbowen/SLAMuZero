from typing import Tuple

import os
from collections import namedtuple
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '0.7'

from absl import app
from absl import flags

import acme 
from acme import wrappers, specs
from mcts.wrappers import MapLoggingWrapper
from acme.agents.tf.mcts import models, ReprMCTS
from acme.agents.tf.mcts.models import simulator, mlp
import sonnet as snt
import dm_env

from env.habitat import dm_make_env_fn
from mcts.networks import ResNetRepresentation
from mcts import agent as mp_agent
from mcts.models import mlp as mp_mlp

from absl import flags


# construct_envs flags
flags.DEFINE_integer('seed', 0, 'seed')
flags.DEFINE_string('task_config', "tasks/pointnav_gibson.yaml", 'task config.')
flags.DEFINE_string('split', 'val', 'split method')
flags.DEFINE_integer('num_processes', 1, 'Number of process to run.')
flags.DEFINE_integer('num_processes_on_first_gpu', 1, 'num processes on first gpu')
flags.DEFINE_integer('num_processes_per_gpu', 0, 'num processes per gpu')
flags.DEFINE_integer('sim_gpu_id', 1, 'simulation gpu id')
flags.DEFINE_integer('max_episode_length', 1000, 'max episode length')
flags.DEFINE_integer('max_training_steps', 20000, 'max training steps')
flags.DEFINE_integer('num_episodes', 1, 'number of episodes')
flags.DEFINE_integer('env_frame_width', 256, 'frame width')
flags.DEFINE_integer('env_frame_height', 256, 'frame height')
flags.DEFINE_float('hfov', 90.0, 'hfov')
# VectorEnv flags
flags.DEFINE_float('camera_height', 1.25, 'camera_height')
flags.DEFINE_integer('visualize', 0, 'to visualize or not')
flags.DEFINE_integer('print_images', 0, 'print images or not')
flags.DEFINE_integer('plot_every', 10, 'print images every plot_every step')
flags.DEFINE_integer('frame_height', 128, 'frame height')
flags.DEFINE_integer('frame_width', 128, 'frame width')
flags.DEFINE_integer('map_resolution', 5, 'map resolution')
flags.DEFINE_integer('map_size_cm', 2400, 'map size in cm')
flags.DEFINE_integer('du_scale', 2, 'du scale')
flags.DEFINE_integer('vision_range', 64, 'vision range')
flags.DEFINE_integer('vis_type', 1, 'visual type')
flags.DEFINE_integer('obstacle_boundary', 5, 'obstacle boundary')
flags.DEFINE_integer('obs_threshold', 1, 'observe threshold')
# reset() flags
flags.DEFINE_integer('randomize_env_every', 5, 'Frequency to randomize environment')
flags.DEFINE_integer('global_downscaling', 2, 'Global downscaling factor')
flags.DEFINE_integer('noisy_actions', 1, 'Whether to use noisy actions (1 for yes, 0 for no)')
flags.DEFINE_integer('noisy_odometry', 1, 'Whether to use noisy odometry (1 for yes, 0 for no)')
flags.DEFINE_integer('num_local_steps', 25, 'Number of local steps')
flags.DEFINE_integer('short_goal_dist', 1, 'Short goal distance')
flags.DEFINE_integer('num_update_per_episode', 50, 'Number of updates per episode')
flags.DEFINE_integer('num_simulations', 50, 'Number of simulations')
flags.DEFINE_integer('num_trajectory', 4, 'Number of trajectories')
flags.DEFINE_integer('sample_per_trajectory', 16, 'Samples per trajectory')
flags.DEFINE_integer('k_steps', 5, 'K steps')
flags.DEFINE_integer('action_width', 10, 'Action width')
flags.DEFINE_integer('action_height', 10, 'Action height')
flags.DEFINE_integer('buffer_capacity', 50, 'Buffer capacity')
flags.DEFINE_integer('n_bootstrapping', 10, 'Number of bootstrapping')
flags.DEFINE_integer('log_interval', 1, 'Log interval')
flags.DEFINE_integer('save_interval', 1, 'Save interval')
flags.DEFINE_integer('save_periodic', 500000, 'Save periodic')
flags.DEFINE_integer('split_key', 16, 'Split key')
flags.DEFINE_float('noise_level', 1.0, 'Noise level')
flags.DEFINE_float('eval_temperature', 0.2, 'Evaluation temperature')
flags.DEFINE_float('collision_threshold', 0.2, 'Collision threshold')
flags.DEFINE_boolean('eval', True, 'Whether to run evaluation')
flags.DEFINE_integer('eval_episodes', 1, 'number of eval episodes')
flags.DEFINE_string('save_trajectory_data', '0', 'Whether to save trajectory data (1 for yes, 0 for no)')
flags.DEFINE_string('dump_location', '../tmp/', 'Location to dump data')
flags.DEFINE_string('exp_name', 'slamuzero_img', 'Name of the experiment')
flags.DEFINE_string('load_model', '0', 'Whether to load a model (1 for yes, 0 for no)')
# agent flags
flags.DEFINE_float('discount', 0.99, 'discount')
flags.DEFINE_list('model_hiddens', [64, 64], 'model mlp hidden sizes')
flags.DEFINE_integer('model_replay_capacity', 1000, 'model\'s replay buffer capacity')
flags.DEFINE_integer('model_batch_size', 16, 'model\'s batch size')
flags.DEFINE_integer('agent_replay_capacity', 10000, 'agent\'s replay buffer capacity')
flags.DEFINE_integer('agent_batch_size', 16, 'agent\'s batch size')
flags.DEFINE_boolean('use_map', True, 'whether to use map module in MCTS')
flags.DEFINE_boolean('nested_action', True, 'whether to use nested action.')
flags.DEFINE_list('eval_output_sizes', [1024, 4096], 'hidden size for evaluation function')
flags.DEFINE_boolean('eval_should_update', False, 'should the actor update during evaluation')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')


# flags.DEFINE_boolean('simulator', True, 'Simulator or learned model?')

FLAGS = flags.FLAGS
FlagTuple = namedtuple('FlagTuple', [name for name in FLAGS.flag_values_dict()])


def convert_flags_to_namedtuple(FLAGS):
    return FlagTuple(**{name: FLAGS[name].value for name in FLAGS if name in FlagTuple._fields})


def make_env_and_model(repr_network) -> Tuple[dm_env.Environment, models.Model]:
  """Create environment and corresponding model (learned or simulator)."""
  flag_values = convert_flags_to_namedtuple(FLAGS)
  env = dm_make_env_fn(flag_values, FLAGS.seed)
  # if FLAGS.simulator:
  #    model = simulator.Simulator(env)
  # else:
  if FLAGS.nested_action and FLAGS.use_map:
    model = mp_mlp.ReprMLPModelNestedAction(
      repr_network,
      specs.make_environment_spec(env),
      replay_capacity=FLAGS.model_replay_capacity,
      batch_size=FLAGS.model_batch_size,
      hidden_sizes=tuple(map(int, FLAGS.model_hiddens)),
      learning_rate=FLAGS.learning_rate,
    )
  else:
    model = mlp.ReprMLPModel(
      repr_network,
      specs.make_environment_spec(env),
      replay_capacity=FLAGS.model_replay_capacity,
      batch_size=FLAGS.model_batch_size,
      hidden_sizes=tuple(map(int, FLAGS.model_hiddens)),
      learning_rate=FLAGS.learning_rate,
    )
  env = wrappers.SinglePrecisionWrapper(env)
  env = MapLoggingWrapper(env)
  return env, model


def make_network(action_spec: specs.DiscreteArray) -> snt.Module:
  if not FLAGS.use_map:  
    from mcts.networks import ResNetEvaluation
    eval_net = ResNetEvaluation(action_spec.num_values, output_sizes=tuple(map(int, FLAGS.eval_output_sizes)))
  else:    
    from mcts.networks import DecEvaluation
    num_values = action_spec.action.num_values if FLAGS.nested_action else action_spec.num_values
    eval_net = DecEvaluation(num_values, output_sizes=tuple(map(int, FLAGS.eval_output_sizes)))
  return eval_net


def main(_):
    repr_network = ResNetRepresentation(frame_height=FLAGS.frame_height, frame_width=FLAGS.frame_width)
    envs, model = make_env_and_model(repr_network)
    environment_spec = specs.make_environment_spec(envs)
    # Create the network and optimizer.
    eval_network = make_network(environment_spec.actions)
    
    optimizer = snt.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    # Construct the agent.
    if FLAGS.use_map:
      args = convert_flags_to_namedtuple(FLAGS)
      if FLAGS.nested_action:
        agent = mp_agent.MapMCTSNestedAction(
          args=args,
          environment_spec=environment_spec,
          model=model,
          repr_network=repr_network,
          eval_network=eval_network,
          optimizer=optimizer,
          discount=FLAGS.discount,
          replay_capacity=FLAGS.agent_replay_capacity,
          n_step=FLAGS.k_steps,
          batch_size=FLAGS.agent_batch_size,
          num_simulations=FLAGS.num_simulations,)
      else:
        agent = mp_agent.MapMCTS(
          args=args,
          environment_spec=environment_spec,
          model=model,
          repr_network=repr_network,
          eval_network=eval_network,
          optimizer=optimizer,
          discount=FLAGS.discount,
          replay_capacity=FLAGS.agent_replay_capacity,
          n_step=FLAGS.k_steps,
          batch_size=FLAGS.agent_batch_size,
          num_simulations=FLAGS.num_simulations,)
    else:
      agent = ReprMCTS(
        environment_spec=environment_spec,
        model=model,
        repr_network=repr_network,
        eval_network=eval_network,
        optimizer=optimizer,
        discount=FLAGS.discount,
        replay_capacity=FLAGS.agent_replay_capacity,
        n_step=FLAGS.k_steps,
        batch_size=FLAGS.agent_batch_size,
        num_simulations=FLAGS.num_simulations,
        )
    # Run the environment loop.
    loop = acme.EnvironmentLoop(envs, agent)
    loop.run(num_episodes=FLAGS.num_episodes) 
    if FLAGS.eval:
      FLAGS.print_images = 1
      if FLAGS.use_map:
        args = convert_flags_to_namedtuple(FLAGS)
        agent.args = args
      flag_values = convert_flags_to_namedtuple(FLAGS)
      eval_env = dm_make_env_fn(flag_values, FLAGS.seed)
      eval_env = wrappers.SinglePrecisionWrapper(eval_env)
      eval_env = MapLoggingWrapper(eval_env)
      agent._actor._adder.reset()  # force reset
      eval_loop = acme.EnvironmentLoop(eval_env, agent, should_update=FLAGS.eval_should_update)
      eval_loop.run(num_episodes=FLAGS.eval_episodes)
       

if __name__ == '__main__':
    app.run(main)