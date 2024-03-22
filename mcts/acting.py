# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A MCTS actor."""

from typing import Optional, Tuple, List, Callable, NamedTuple

import os 
import math 
import functools
import acme
from acme import adders
from acme import specs
from acme.agents.tf.mcts import models
from acme.agents.tf.mcts.acting import MCTSActor
from acme.agents.tf.mcts import search
from acme.agents.tf.mcts import types
from acme.tf import variable_utils as tf2_variable_utils
from acme.tf import utils as tf2_utils

import dm_env
import numpy as np
from scipy import special
import sonnet as snt
import tensorflow as tf
import skimage
import matplotlib.pyplot as plt 

from . import utils as mp_utils
from . import search as mp_search
from . import types as mp_types


class MapMCTSActor(MCTSActor):
  """Executes a policy- and value-network guided MCTS search with
    representation for raw observation(MuZero setting) and internally builds a map.
  """

  _prev_timestep: dm_env.TimeStep

  def __init__(
      self,
      args,
      environment_spec: specs.EnvironmentSpec,
      model: models.Model,
      repr_network: snt.Module,
      eval_network: snt.Module,
      discount: float,
      num_simulations: int,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
  ):
    self.args = args 

    # Internalize components: model, network, data sink and variable source.
    self._model = model
    self._repr_network = tf.function(repr_network)
    self._eval_network = tf.function(eval_network)
    self._map_func = tf.function(eval_network.decode)
    self._variable_client = variable_client
    self._adder = adder

    # Internalize hyperparameters.
    self._num_actions = environment_spec.actions.num_values
    self._num_simulations = num_simulations
    self._actions = list(range(self._num_actions))
    self._discount = discount

    # We need to save the policy so as to add it to replay on the next step.
    self._probs = np.ones(
        shape=(self._num_actions,), dtype=np.float32) / self._num_actions
    # Also the map 
    self._maps = np.zeros_like((self.args.vision_range, self.args.vision_range, 2), dtype=np.float32)
    self._pos = np.zeros_like((3,), dtype=np.float32)

    self._init_obs_dims()

  def _forward(
      self, hidden_state: types.Observation) -> Tuple[types.Probs, types.Value]:
    """Performs a forward pass of the policy-value network."""
    logits, value = self._eval_network(tf.expand_dims(hidden_state, axis=0))

    # Convert to numpy & take softmax.
    logits = logits.numpy().squeeze(axis=0)
    value = value.numpy().item()
    probs = special.softmax(logits)

    return probs, value

  def select_action(self, observation: types.Observation) -> types.Action:
    """Computes the agent's policy via MCTS."""
    if self._model.needs_reset:
      self._model.reset(observation)

    _hidden_state = self._repr_network(tf.expand_dims(observation, axis=0))
    hidden_state = tf.squeeze(_hidden_state, axis=0)
    _maps, _pos = self._map_func(_hidden_state)
    _maps = tf.squeeze(_maps, axis=0).numpy()
    _pos = tf.squeeze(_pos, axis=0).numpy()
    
    # Compute a fresh MCTS plan.
    root = search.mcts(
        hidden_state,
        model=self._model,
        search_policy=search.puct,
        evaluation=self._forward,
        num_simulations=self._num_simulations,
        num_actions=self._num_actions,
        discount=self._discount,
    )

    # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
    probs = search.visit_count_policy(root)
    action = np.int32(np.random.choice(self._actions, p=probs))

    # Save the policy probs so that we can add them to replay in `observe()`.
    self._probs = probs.astype(np.float32)
    self._maps = self._get_map_target(observation)
    self._pos = observation[self._p3: self._p4]

    return action

  def update(self, wait: bool = False):
    """Fetches the latest variables from the variable source, if needed."""
    if self._variable_client:
      self._variable_client.update(wait)

  def observe_first(self, timestep: dm_env.TimeStep):
    self._prev_timestep = timestep
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.Action, next_timestep: dm_env.TimeStep):
    """Updates the agent's internal model and adds the transition to replay."""
    self._model.update(self._prev_timestep, action, next_timestep)

    self._prev_timestep = next_timestep

    if self._adder:
      self._adder.add(action, next_timestep, extras={'pi': self._probs, 'map': self._maps, 'pos': self._pos})

  def _init_obs_dims(self):
    """Identical to environment definition."""
    self._num_channels = 3
    self._rgb_dim = np.prod((self.args.frame_height, self.args.frame_width, self._num_channels))
    self._fp_proj_dim = self._fp_explored_dim = np.prod((self.args.vision_range, self.args.vision_range))
    self._sensor_pose_dim = self._pose_err_dim = 3 
    self._exp_reward_dim = self._exp_ratio_dim = 1
    self._map_dim = (self.args.map_size_cm // self.args.map_resolution) ** 2
    self._p1 = self._rgb_dim + self._fp_proj_dim
    self._p2 = self._p1 + self._fp_explored_dim
    self._p3 = self._p2 + self._sensor_pose_dim
    self._p4 = self._p3 + self._pose_err_dim
    self._p5 = self._p4 + self._exp_reward_dim
    self._p6 = self._p5 + self._exp_ratio_dim

  def _get_map_target(self, observation: types.Observation):
    gt_map = observation[self._rgb_dim: self._p1].reshape(self.args.vision_range, self.args.vision_range)
    gt_explore = observation[self._p1: self._p2].reshape(self.args.vision_range, self.args.vision_range)
    gt_maps = np.stack((gt_map, gt_explore), axis=-1).astype(np.float32)
    return gt_maps


class MapMCTSActorNestedAction(MapMCTSActor):
  def __init__(
      self,
      args,
      environment_spec: specs.EnvironmentSpec,
      model: models.Model,
      repr_network: snt.Module,
      eval_network: snt.Module,
      discount: float,
      num_simulations: int,
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[tf2_variable_utils.VariableClient] = None,
  ):
    self.args = args 

    # Internalize components: model, network, data sink and variable source.
    self._model = model
    self._repr_network = tf.function(repr_network)
    self._eval_network = tf.function(eval_network)
    self._map_func = tf.function(eval_network.decode)
    self._variable_client = variable_client
    self._adder = adder

    # Internalize hyperparameters.
    self._num_actions = environment_spec.actions.action.num_values
    self._num_simulations = num_simulations
    self._actions = list(range(self._num_actions))
    self._discount = discount

    # We need to save the policy so as to add it to replay on the next step.
    self._probs = np.ones(
        shape=(self._num_actions,), dtype=np.float32) / self._num_actions
    # Also the map 
    self._maps = np.zeros_like((self.args.vision_range, self.args.vision_range, 2), dtype=np.float32)
    self._pos = np.zeros_like((3,), dtype=np.float32)

    self._init_obs_dims()

  def select_action(self, observation: types.Observation) -> mp_types.ActionExtras:
    """Computes the agent's policy via MCTS."""
    if self._model.needs_reset:
      self._model.reset(observation)

    _hidden_state = self._repr_network(tf.expand_dims(observation, axis=0))
    hidden_state = tf.squeeze(_hidden_state, axis=0)
    _maps, _pos = self._map_func(_hidden_state)
    _maps = tf.squeeze(_maps, axis=0).numpy().astype(np.float32)
    _pos = tf.squeeze(_pos, axis=0).numpy().astype(np.float32)

    if self._timestamp % self.args.num_local_steps == 0:
      # Compute a fresh MCTS plan.
      root = mp_search.mcts_nested_action(
          hidden_state,
          model=self._model,
          search_policy=search.puct,
          evaluation=self._forward,
          num_simulations=self._num_simulations,
          num_actions=self._num_actions,
          discount=self._discount,
      )

      # The agent's policy is softmax w.r.t. the *visit counts* as in AlphaZero.
      probs = search.visit_count_policy(root)
      action = np.int32(np.random.choice(self._actions, p=probs))
      self._prev_action = action
      # Save the policy probs so that we can add them to replay in `observe()`.
      self._probs = probs.astype(np.float32)
    
    self._maps = self._get_map_target(observation)
    self._pos = observation[self._p3: self._p4]

    self._timestamp += 1

    return mp_types.ActionExtras(
      action=self._prev_action, 
      map_prediction=_maps[...,0], 
      explored_prediction=_maps[...,1],
      pos_prediction=_pos)
  
  def observe_first(self, timestep: dm_env.TimeStep):
    self._prev_timestep = timestep
    self._timestamp = 0
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.Action, next_timestep: dm_env.TimeStep):
    """Updates the agent's internal model and adds the transition to replay."""
    if (self._timestamp - 1) % self.args.num_local_steps == 0 or next_timestep.last():
      self._model.update(self._prev_timestep, action, next_timestep)

    self._prev_timestep = next_timestep

    if self._adder and ((self._timestamp - 1) % self.args.num_local_steps == 0 or next_timestep.last()):
      self._adder.add(action, next_timestep, extras={'pi': self._probs, 'map': self._maps, 'pos': self._pos})
  