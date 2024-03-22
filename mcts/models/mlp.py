from typing import Optional, Tuple, Dict, List

from acme import specs
from acme.agents.tf.mcts import types
from acme.agents.tf.mcts.models import base
from acme.tf import utils as tf2_utils

from bsuite.baselines.utils import replay
import dm_env
import numpy as np
from scipy import special
import sonnet as snt
import tensorflow as tf

from mcts import types as mp_types
    

class MLPTransitionModelNestedAction(snt.Module):
  """This uses MLPs to model (s, a) -> (r, d, s')."""

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      hidden_sizes: Tuple[int, ...],
  ):
    super(MLPTransitionModelNestedAction, self).__init__(name='mlp_transition_model')

    # Get num actions/observation shape.
    self._num_actions = environment_spec.actions.action.num_values
    self._input_shape = environment_spec.observations.shape
    self._flat_shape = int(np.prod(self._input_shape))

    # Prediction networks.
    self._state_network = snt.Sequential([
        snt.nets.MLP(hidden_sizes + (self._flat_shape,)),
        snt.Reshape(self._input_shape)
    ])
    self._reward_network = snt.Sequential([
        snt.nets.MLP(hidden_sizes + (1,)),
        lambda r: tf.squeeze(r, axis=-1),
    ])
    self._discount_network = snt.Sequential([
        snt.nets.MLP(hidden_sizes + (1,)),
        lambda d: tf.squeeze(d, axis=-1),
    ])

  def __call__(self, state: tf.Tensor,
               action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    embedded_state = snt.Flatten()(state)
    embedded_action = tf.one_hot(action, depth=self._num_actions)
    embedding = tf.concat([embedded_state, embedded_action], axis=-1)

    # Predict the next state, reward, and termination.
    next_state = self._state_network(embedding)
    reward = self._reward_network(embedding)
    discount_logits = self._discount_network(embedding)

    return next_state, reward, discount_logits


class MLPModelNestedAction(base.Model):
  """A simple environment model."""

  _checkpoint: types.Observation
  _state: types.Observation

  def __init__(
      self,
      environment_spec: specs.EnvironmentSpec,
      replay_capacity: int,
      batch_size: int,
      hidden_sizes: Tuple[int, ...],
      learning_rate: float = 1e-3,
      terminal_tol: float = 1e-3,
  ):
    self._obs_spec = environment_spec.observations
    self._action_spec = environment_spec.actions
    # Hyperparameters.
    self._batch_size = batch_size
    self._terminal_tol = terminal_tol

    # Modelling
    self._replay = replay.Replay(replay_capacity)
    self._transition_model = MLPTransitionModelNestedAction(environment_spec, hidden_sizes)
    self._optimizer = snt.optimizers.Adam(learning_rate)
    self._forward = tf.function(self._transition_model)
    tf2_utils.create_variables(
        self._transition_model, [self._obs_spec, self._action_spec])
    self._variables = self._transition_model.trainable_variables

    # Model state.
    self._needs_reset = True

  @tf.function
  def _step(
      self,
      o_t: tf.Tensor,
      a_t: tf.Tensor,
      r_t: tf.Tensor,
      d_t: tf.Tensor,
      o_tp1: tf.Tensor,
  ) -> tf.Tensor:

    with tf.GradientTape() as tape:
      next_state, reward, discount = self._transition_model(o_t, a_t)

      state_loss = tf.square(next_state - o_tp1)
      reward_loss = tf.square(reward - r_t)
      discount_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_t, discount)

      loss = sum([
          tf.reduce_mean(state_loss),
          tf.reduce_mean(reward_loss),
          tf.reduce_mean(discount_loss),
      ])

    gradients = tape.gradient(loss, self._variables)
    self._optimizer.apply(gradients, self._variables)

    return loss

  def step(self, action: mp_types.ActionExtras):
    # Reset if required.
    if self._needs_reset:
      raise ValueError('Model must be reset with an initial timestep.')

    # Step the model.
    state, action = tf2_utils.add_batch_dim([self._state, action.action])
    new_state, reward, discount_logits = [
        x.numpy().squeeze(axis=0) for x in self._forward(state, action)
    ]
    discount = special.softmax(discount_logits)

    # Save the resulting state for the next step.
    self._state = new_state

    # We threshold discount on a given tolerance.
    if discount < self._terminal_tol:
      self._needs_reset = True
      return dm_env.termination(reward=reward, observation=self._state.copy())
    return dm_env.transition(reward=reward, observation=self._state.copy())

  def reset(self, initial_state: Optional[types.Observation] = None):
    if initial_state is None:
      raise ValueError('Model must be reset with an initial state.')
    # We reset to an initial state that we are explicitly given.
    # This allows us to handle environments with stochastic resets (e.g. Catch).
    self._state = initial_state.copy()
    self._needs_reset = False
    return dm_env.restart(self._state)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: mp_types.ActionExtras,
      next_timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    # Add the true transition to replay.
    transition = [
        timestep.observation,
        action.action,
        next_timestep.reward,
        next_timestep.discount,
        next_timestep.observation,
    ]
    self._replay.add(transition)

    # Step the model to generate a synthetic transition.
    ts = self.step(action)

    # Copy the *true* state on update.
    self._state = next_timestep.observation.copy()

    if ts.last() or next_timestep.last():
      # Model believes that a termination has happened.
      # This will result in a crash during planning if the true environment
      # didn't terminate here as well. So, we indicate that we need a reset.
      self._needs_reset = True

    # Sample from replay and do SGD.
    if self._replay.size >= self._batch_size:
      batch = self._replay.sample(self._batch_size)
      self._step(*batch)

    return ts

  def save_checkpoint(self):
    if self._needs_reset:
      raise ValueError('Cannot save checkpoint: model must be reset first.')
    self._checkpoint = self._state.copy()

  def load_checkpoint(self):
    self._needs_reset = False
    self._state = self._checkpoint.copy()

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._obs_spec

  @property
  def needs_reset(self) -> bool:
    return self._needs_reset


class ReprMLPTransitionModelNestedAction(snt.Module):
  """This uses MLPs to model (s, a) -> (r, d, s')."""

  def __init__(
      self,
      repr_spec,
      environment_spec: specs.EnvironmentSpec,
      hidden_sizes: Tuple[int, ...],
  ):
    super(ReprMLPTransitionModelNestedAction, self).__init__(name='mlp_transition_model')

    # Get num actions/observation shape.
    self._num_actions = environment_spec.actions.action.num_values
    self._input_shape = repr_spec.shape
    self._flat_shape = int(np.prod(self._input_shape))

    # Prediction networks.
    self._state_network = snt.Sequential([
        snt.nets.MLP(hidden_sizes + (self._flat_shape,)),
        snt.Reshape(self._input_shape)
    ])
    self._reward_network = snt.Sequential([
        snt.nets.MLP(hidden_sizes + (1,)),
        lambda r: tf.squeeze(r, axis=-1),
    ])
    self._discount_network = snt.Sequential([
        snt.nets.MLP(hidden_sizes + (1,)),
        lambda d: tf.squeeze(d, axis=-1),
    ])

  def __call__(self, state: tf.Tensor,
               action: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    embedded_state = snt.Flatten()(state)
    embedded_action = tf.one_hot(action, depth=self._num_actions)
    embedding = tf.concat([embedded_state, embedded_action], axis=-1)

    # Predict the next state, reward, and termination.
    next_state = self._state_network(embedding)
    reward = self._reward_network(embedding)
    discount_logits = self._discount_network(embedding)

    return next_state, reward, discount_logits
  

class ReprMLPModelNestedAction(MLPModelNestedAction):
  def __init__(
      self,
      repr_network: snt.Module,
      environment_spec: specs.EnvironmentSpec,
      replay_capacity: int,
      batch_size: int,
      hidden_sizes: Tuple[int, ...],
      learning_rate: float = 1e-3,
      terminal_tol: float = 1e-3,
  ):
    repr_output_spec = tf2_utils.create_variables(repr_network, [environment_spec.observations])
    self._repr_network = tf.function(repr_network)
    self._obs_spec = environment_spec.observations
    self._action_spec = environment_spec.actions
    # Hyperparameters.
    self._batch_size = batch_size
    self._terminal_tol = terminal_tol

    # Modelling
    self._replay = replay.Replay(replay_capacity)
    self._transition_model = ReprMLPTransitionModelNestedAction(repr_output_spec, environment_spec, hidden_sizes)
    self._optimizer = snt.optimizers.Adam(learning_rate)
    self._forward = tf.function(self._transition_model)
    tf2_utils.create_variables(
        self._transition_model, [repr_output_spec, self._action_spec.action])
    self._variables = self._transition_model.trainable_variables + repr_network.trainable_variables

    # Model state.
    self._needs_reset = True

  @tf.function
  def _step(
      self,
      o_t: tf.Tensor,
      a_t: tf.Tensor,
      r_t: tf.Tensor,
      d_t: tf.Tensor,
      o_tp1: tf.Tensor,
  ) -> tf.Tensor:

    with tf.GradientTape() as tape:
      next_state, reward, discount = self._transition_model(self._repr_network(o_t), a_t)

      state_loss = tf.square(next_state - tf.stop_gradient(self._repr_network(o_tp1)))
      reward_loss = tf.square(reward - r_t)
      discount_loss = tf.nn.sigmoid_cross_entropy_with_logits(d_t, discount)

      loss = sum([
          tf.reduce_mean(state_loss),
          tf.reduce_mean(reward_loss),
          tf.reduce_mean(discount_loss),
      ])

    gradients = tape.gradient(loss, self._variables)
    self._optimizer.apply(gradients, self._variables)

    return loss
  
  def update(
      self,
      timestep: dm_env.TimeStep,
      action: mp_types.ActionExtras,
      next_timestep: dm_env.TimeStep,
  ) -> dm_env.TimeStep:
    # Add the true transition to replay.
    transition = [
        timestep.observation,
        action.action,
        next_timestep.reward,
        next_timestep.discount,
        next_timestep.observation,
    ]
    self._replay.add(transition)

    # Step the model to generate a synthetic transition.
    ts = self.step(action)
    
    # Copy the *true* state's representation on update.
    self._state = self._repr_network(tf.expand_dims(next_timestep.observation, axis=0)).numpy().squeeze(0).copy()

    if ts.last() or next_timestep.last():
      # Model believes that a termination has happened.
      # This will result in a crash during planning if the true environment
      # didn't terminate here as well. So, we indicate that we need a reset.
      self._needs_reset = True

    # Sample from replay and do SGD.
    if self._replay.size >= self._batch_size:
      batch = self._replay.sample(self._batch_size)
      self._step(*batch)

    return ts
  
  def reset(self, initial_state: Optional[types.Observation] = None):
    if initial_state is None:
      raise ValueError('Model must be reset with an initial state.')
    _initial_state = self._repr_network(tf.expand_dims(initial_state, axis=0)).numpy().squeeze(0)
    
    return super().reset(_initial_state)
  