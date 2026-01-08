from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .game_mode import GameType

if TYPE_CHECKING:
    from ._riichienv import RiichiEnv


class GameMode(ABC):
    def __init__(self, target_score: int = 30000, max_extension_field: int = 0, tobi: bool = True):
        self.target_score = target_score
        self.max_extension_field = max_extension_field
        self.tobi = tobi

    def is_tobi(self, env: "RiichiEnv") -> bool:
        return self.tobi and any(s < 0 for s in env.scores())

    @abstractmethod
    def is_game_over(
        self, env: "RiichiEnv", is_renchan: bool, is_draw: bool = False, is_midway_draw: bool = False
    ) -> bool:
        """Determines if the game should end after the current kyoku."""
        pass

    def get_next_kyoku_params(self, env: "RiichiEnv", is_renchan: bool, was_draw: bool = False) -> dict:
        """Returns the parameters for the next kyoku."""
        next_honba = env._custom_honba + 1 if (is_renchan or was_draw) else 0

        if is_renchan:
            return {
                "oya": env.oya,
                "bakaze": env._custom_round_wind,
                "honba": next_honba,
                "kyotaku": env.riichi_sticks,
            }
        else:
            next_oya = (env.oya + 1) % 4
            next_bakaze = env._custom_round_wind
            if next_oya == 0:
                next_bakaze += 1
            return {
                "oya": next_oya,
                "bakaze": next_bakaze,
                "honba": next_honba,
                "kyotaku": env.riichi_sticks,
            }


class OneKyokuGameMode(GameMode):
    def is_game_over(
        self, env: "RiichiEnv", is_renchan: bool, is_draw: bool = False, is_midway_draw: bool = False
    ) -> bool:
        if self.is_tobi(env):
            return True
        # Default behavior: One Kyoku always ends after one round regardless of renchan.
        return True


class StandardGameMode(GameMode):
    def __init__(self, end_field: int, target_score: int, max_extension_field: int, tobi: bool = True):
        super().__init__(target_score, max_extension_field, tobi)
        self.end_field = end_field  # 0: East, 1: South, 2: West

    def is_game_over(  # noqa: PLR0911
        self, env: "RiichiEnv", is_renchan: bool, is_draw: bool = False, is_midway_draw: bool = False
    ) -> bool:
        if self.is_tobi(env):
            return True
        scores = env.scores()
        max_score = max(scores)
        current_field = env._custom_round_wind
        current_oya = env.oya

        # Last round of the standard game (e.g. South 4)
        is_last_round = (current_field == self.end_field) and (current_oya == 3)

        if is_renchan:
            if is_midway_draw:
                return False  # No yame on abortive draws

            # Agari-yame / Tenpai-yame logic if oya is 1st and >= target_score
            if is_last_round or current_field > self.end_field:
                # Rank 1 is the top player (indices are 1-based)
                if env.ranks()[current_oya] == 1 and scores[current_oya] >= self.target_score:
                    return True
            return False

        # Not renchan.
        if is_last_round:
            if max_score >= self.target_score:
                return True
            # Enter extension
            return False

        if current_field > self.end_field:
            # In extension (Sudden Death / V-Goal)
            if max_score >= self.target_score:
                return True
            if current_field == self.max_extension_field and current_oya == 3:
                return True
            return False

        return False


class TonpuuGameMode(StandardGameMode):
    def __init__(self, target_score: int = 30000, tobi: bool = True):
        super().__init__(end_field=0, target_score=target_score, max_extension_field=1, tobi=tobi)


class HanchanGameMode(StandardGameMode):
    def __init__(self, target_score: int = 30000, tobi: bool = True):
        super().__init__(end_field=1, target_score=target_score, max_extension_field=2, tobi=tobi)


class SuddenDeathIkkyokuGameMode(OneKyokuGameMode):
    def __init__(self, target_score: int = 30000):
        super().__init__(target_score=target_score)

    def is_game_over(
        self, env: "RiichiEnv", is_renchan: bool, is_draw: bool = False, is_midway_draw: bool = False
    ) -> bool:
        if any(s >= self.target_score for s in env.scores()):
            return True
        return False


def get_game_mode(game_type: GameType) -> GameMode:
    if game_type in [GameType.YON_IKKYOKU, GameType.SAN_IKKYOKU]:
        return OneKyokuGameMode(target_score=0, tobi=True)
    elif game_type == GameType.YON_TONPUSEN:
        return TonpuuGameMode(tobi=True)
    elif game_type == GameType.SAN_TONPUSEN:
        return TonpuuGameMode(tobi=False)  # Example: San-nin might have different default
    elif game_type == GameType.YON_HANCHAN:
        return HanchanGameMode(tobi=True)
    elif game_type == GameType.SAN_HANCHAN:
        return HanchanGameMode(tobi=False)
    return OneKyokuGameMode(target_score=0, tobi=True)
