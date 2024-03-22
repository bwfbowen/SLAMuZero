from typing import Optional
from acme.agents.tf.mcts import learning
import sonnet as snt
import tensorflow as tf
from acme.utils import counting, loggers

class MapMZLearner(learning.MZLearner):
  def __init__(
        self, 
        repr_network: snt.Module, 
        eval_network: snt.Module, 
        optimizer: snt.Optimizer, 
        dataset: tf.data.Dataset, 
        discount: float, 
        logger: Optional[loggers.Logger] = None,
        counter: Optional[counting.Counter] = None,
        checkpoint: bool = True, 
        save_directory: str = '~/acme',
        csv_directory: str = '~/acme',
    ):
    super().__init__(repr_network, eval_network, optimizer, dataset, discount, logger, counter, checkpoint, save_directory)
    self._csv_logger = loggers.CSVLogger(directory_or_file=csv_directory, label='learner_csv_log') if csv_directory else None 
    
  @tf.function
  def _step(self) -> tf.Tensor:
    """Do a step of SGD on the loss."""
    inputs = next(self._iterator)
    o_t, _, r_t, d_t, o_tp1, extras = inputs.data
    pi_t, map_t, pos_t = extras['pi'], extras['map'], extras['pos']

    with tf.GradientTape() as tape:
      # representation
      h_t = self._repr_network(o_t)
      h_tp1 = self._repr_network(o_tp1)
      # Compute map for observation
      mpred_t, ppred_t = self._eval_network.decode(h_t)
      # Forward the network on the two states in the transition.
      logits, value = self._eval_network(h_t)
      _, target_value = self._eval_network(h_tp1)
      target_value = tf.stop_gradient(target_value)

      # Value loss is simply on-policy TD learning.
      value_loss = tf.square(r_t + self._discount * d_t * target_value - value)

      # Policy loss distills MCTS policy into the policy network.
      policy_loss = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=pi_t)

      # Mapping loss
      map_loss = tf.keras.metrics.binary_crossentropy(
        tf.reshape(map_t, [tf.shape(map_t)[0], -1]), 
        tf.reshape(mpred_t, [tf.shape(mpred_t)[0], -1]),
        from_logits=True)
      
      pos_loss = tf.nn.l2_loss(ppred_t - pos_t)

      # Compute gradients.
      loss = tf.reduce_mean(value_loss + policy_loss + map_loss)
      
      gradients = tape.gradient(loss, self._eval_network.trainable_variables + self._repr_network.trainable_variables)
    
    log_map_loss = tf.reduce_mean(map_loss)
    log_pos_loss = tf.reduce_mean(pos_loss)
    self._optimizer.apply(gradients, self._eval_network.trainable_variables + self._repr_network.trainable_variables)

    return loss, log_map_loss, log_pos_loss 
  
  def step(self):
    """Does a step of SGD and logs the results."""
    loss, log_map_loss, log_pos_loss = self._step()
    _log = {'loss': loss, 'map error': log_map_loss, 'pos error': log_pos_loss}
    self._logger.write(_log)
    self._csv_logger.write(_log)
    if self._checkpointer: self._checkpointer.save()
    if self._snapshotter: self._snapshotter.save()

