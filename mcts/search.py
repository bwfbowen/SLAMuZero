from typing import Callable, Dict, Tuple, List, Optional, NamedTuple

from acme.agents.tf.mcts import models
from acme.agents.tf.mcts import types
from acme.agents.tf.mcts import search
import tensorflow as tf 
import numpy as np

from mcts import types as mp_types

TransitionProcessor = Callable[[types.Observation, types.Observation, types.Action], types.Observation]


def action_aware_mcts(
    observation: types.Observation,
    model: models.Model,
    search_policy: search.SearchPolicy,
    evaluation: types.EvaluationFn,
    transition_processor: TransitionProcessor,
    num_simulations: int,
    num_actions: int,
    discount: float = 1.,
    dirichlet_alpha: float = 1,
    exploration_fraction: float = 0.,
    extras: Optional[NamedTuple] = None,
) -> search.Node:
  """Does Monte Carlo tree search (MCTS), with evaluation that is aware of action taken."""

  # Evaluate the prior policy for this state.
  if extras is not None:
    eval_obs = transition_processor(observation, extras.prev_observation, extras.action)
  else:
    eval_obs = observation
  prior, value = evaluation(eval_obs)
  assert prior.shape == (num_actions,)

  # Add exploration noise to the prior.
  noise = np.random.dirichlet(alpha=[dirichlet_alpha] * num_actions)
  prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

  # Create a fresh tree search.
  root = search.Node()
  root.expand(prior)

  # Save the model state so that we can reset it for each simulation.
  model.save_checkpoint()
  for _ in range(num_simulations):
    # Start a new simulation from the top.
    trajectory = [root]
    node = root

    # Generate a trajectory.
    timestep = None
    prev_obs = tf.squeeze(observation, axis=0)
    while node.children:
      # Select an action according to the search policy.
      action = search_policy(node)

      # Point the node at the corresponding child.
      node = node.children[action]

      # Step the simulator and add this timestep to the node.
      if timestep is not None:
        prev_obs = timestep.observation
      timestep = model.step(action)
      node.reward = timestep.reward or 0.
      node.terminal = timestep.last()
      trajectory.append(node)

    if timestep is None:
      raise ValueError('Generated an empty rollout; this should not happen.')

    # Calculate the bootstrap for leaf nodes.
    if node.terminal:
      # If terminal, there is no bootstrap value.
      value = 0.
    else:
      # Otherwise, bootstrap from this node with our value function.
      eval_obs = transition_processor(tf.expand_dims(timestep.observation, axis=0), tf.expand_dims(prev_obs, axis=0), action)
      prior, value = evaluation(eval_obs)

      # We also want to expand this node for next time.
      node.expand(prior)

    # Load the saved model state.
    model.load_checkpoint()

    # Monte Carlo back-up with bootstrap from value function.
    ret = value
    while trajectory:
      # Pop off the latest node in the trajectory.
      node = trajectory.pop()

      # Accumulate the discounted return
      ret *= discount
      ret += node.reward

      # Update the node.
      node.total_value += ret
      node.visit_count += 1

  return root


def mcts_nested_action(
    observation: types.Observation,
    model: models.Model,
    search_policy: search.SearchPolicy,
    evaluation: types.EvaluationFn,
    num_simulations: int,
    num_actions: int,
    discount: float = 1.,
    dirichlet_alpha: float = 1,
    exploration_fraction: float = 0.,
) -> search.Node:
  """Does Monte Carlo tree search (MCTS), AlphaZero style."""

  # Evaluate the prior policy for this state.
  prior, value = evaluation(observation)
  assert prior.shape == (num_actions,)

  # Add exploration noise to the prior.
  noise = np.random.dirichlet(alpha=[dirichlet_alpha] * num_actions)
  prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

  # Create a fresh tree search.
  root = search.Node()
  root.expand(prior)

  # Save the model state so that we can reset it for each simulation.
  model.save_checkpoint()
  for _ in range(num_simulations):
    # Start a new simulation from the top.
    trajectory = [root]
    node = root

    # Generate a trajectory.
    timestep = None
    while node.children:
      # Select an action according to the search policy.
      action = search_policy(node)

      # Point the node at the corresponding child.
      node = node.children[action]

      # Step the simulator and add this timestep to the node.
      timestep = model.step(mp_types.ActionExtras(action=action))
      node.reward = timestep.reward or 0.
      node.terminal = timestep.last()
      trajectory.append(node)

    if timestep is None:
      raise ValueError('Generated an empty rollout; this should not happen.')

    # Calculate the bootstrap for leaf nodes.
    if node.terminal:
      # If terminal, there is no bootstrap value.
      value = 0.
    else:
      # Otherwise, bootstrap from this node with our value function.
      prior, value = evaluation(timestep.observation)

      # We also want to expand this node for next time.
      node.expand(prior)

    # Load the saved model state.
    model.load_checkpoint()

    # Monte Carlo back-up with bootstrap from value function.
    ret = value
    while trajectory:
      # Pop off the latest node in the trajectory.
      node = trajectory.pop()

      # Accumulate the discounted return
      ret *= discount
      ret += node.reward

      # Update the node.
      node.total_value += ret
      node.visit_count += 1

  return root