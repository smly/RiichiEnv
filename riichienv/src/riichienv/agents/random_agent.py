import random

from riichienv import Action, Observation


class RandomAgent:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def act(self, obs: Observation) -> Action:
        """
        Returns a valid Action object (DISCARD, RON, etc).
        For now, mostly discards randomly from legal moves.
        """
        return self._rng.choice(obs.legal_actions())
