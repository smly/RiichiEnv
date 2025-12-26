from typing import Optional
import random

from riichienv.env import Observation
from riichienv.action import Action


class RandomAgent:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        
    def act(self, obs: Observation) -> Action:
        """
        Returns a valid Action object (DISCARD, RON, etc).
        For now, mostly discards randomly from legal moves.
        """
        legal = obs.legal_actions()
        return self._rng.choice(legal)
