from typing import Optional

from acme.utils import loggers
from acme.wrappers import EnvironmentWrapper
import dm_env


class MapLoggingWrapper(EnvironmentWrapper):
    def __init__(self, 
                 environment: dm_env.Environment,
                 logger: Optional[loggers.Logger] = None,
                 csv_directory: str = '~/acme'
    ):
        super().__init__(environment)
        self._logger = logger or loggers.TerminalLogger()
        self._csv_logger = loggers.CSVLogger(directory_or_file=csv_directory, label='env_csv_log') if csv_directory else None 

    def step(self, action) -> dm_env.TimeStep:
        transition = super().step(action)
        exp_reward = transition.reward
        exp_ratio = exp_reward / 0.02 * 10000 / 25 / self._total_area
        self._total_ratio += exp_ratio
        _log = {'explore reward': exp_reward, 'explore ratio': self._total_ratio}
        self._logger.write(_log)
        if self._csv_logger: self._csv_logger.write(_log)
        return transition
    
    def reset(self) -> dm_env.TimeStep:
        _timestep = super().reset()
        self._total_ratio = 0.
        self._total_area = self._environment.explorable_map.sum()
        return _timestep
    