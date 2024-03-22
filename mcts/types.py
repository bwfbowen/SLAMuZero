from typing import List, NamedTuple
import numpy as np 


class PlannerInput(NamedTuple):
    goal: List[int]
    map_prediction: np.ndarray = np.array([], dtype=np.float32)
    explored_prediction: np.ndarray = np.array([], dtype=np.float32) 
    pos_prediction: np.ndarray = np.array([], dtype=np.float32)


class ActionExtras(NamedTuple):
    action: int
    map_prediction: np.ndarray = np.array([], dtype=np.float32)
    explored_prediction: np.ndarray = np.array([], dtype=np.float32) 
    pos_prediction: np.ndarray = np.array([], dtype=np.float32) 
