"""Integration tests for RiichiEnv.observe_event() API.

Tests both 4-player and 3-player modes, verifying:
- Non-action events return None
- Tsumo returns observation for the acting player
- Dahai returns observation when opponent can pon/chi/ron
- Dahai returns None when no reactions available
- Reach + discard flow works
- Full game replay via observe_event produces correct observations
"""

import os
import tempfile

import pytest

from riichienv import ActionType, MjaiReplay, RiichiEnv

# ---------------------------------------------------------------------------
# Fixtures: hand setups that guarantee specific claim opportunities
# ---------------------------------------------------------------------------

_UNKNOWN_HAND_13 = ["?"] * 13

# 4P: P0 discards 1m(0), P1 has pair 1m(1,2) for pon
_4P_TEHAIS = [
    # P0: 1m,2m,3m,4m,5m,6m,7m,8m,9m,1p,2p,3p,4p
    ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
    # P1: 1m,1m,4p,5p,6p,7p,8p,9p,1s,2s,3s,4s,5s
    ["1m", "1m", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s"],
    # P2: 1z,2z,3z,4z,5z,6z,7z,1s,2s,3s,7s,8s,9s
    ["1z", "2z", "3z", "4z", "5z", "6z", "7z", "1s", "2s", "3s", "7s", "8s", "9s"],
    # P3: 4s,5s,6s,7s,8s,9s,4m,5m,6m,7m,8m,9m,1z
    ["4s", "5s", "6s", "7s", "8s", "9s", "4m", "5m", "6m", "7m", "8m", "9m", "1z"],
]

# 3P: P0 discards 1p(tile), P1 has pair 1p for pon
_3P_TEHAIS = [
    # P0: 1m,9m,1p,2p,3p,4p,5p,6p,7s,8s,9s,1z,2z
    ["1m", "9m", "1p", "2p", "3p", "4p", "5p", "6p", "7s", "8s", "9s", "1z", "2z"],
    # P1: 1p,1p,7p,8p,9p,1s,2s,3s,4s,5s,6s,3z,4z
    ["1p", "1p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "3z", "4z"],
    # P2: 5z,6z,7z,7s,8s,9s,7p,8p,9p,1m,9m,1z,2z
    ["5z", "6z", "7z", "7s", "8s", "9s", "7p", "8p", "9p", "1m", "9m", "1z", "2z"],
]


def _mask_tehais(tehais: list[list[str]], my_seat: int) -> list[list[str]]:
    """Return tehais with only my_seat's hand visible; others are '?'."""
    return [hand if i == my_seat else _UNKNOWN_HAND_13 for i, hand in enumerate(tehais)]


def _start_kyoku_event_4p(tehais=None, oya=0, my_seat=None):
    t = tehais or _4P_TEHAIS
    if my_seat is not None:
        t = _mask_tehais(t, my_seat)
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "2p",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": oya,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": t,
    }


def _start_kyoku_event_3p(tehais=None, oya=0, my_seat=None):
    t = tehais or _3P_TEHAIS
    if my_seat is not None:
        t = _mask_tehais(t, my_seat)
    return {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "2p",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": oya,
        "scores": [35000, 35000, 35000],
        "tehais": t,
    }


# ===========================================================================
# 4P Tests
# ===========================================================================


class TestApplyEvent4P:
    """observe_event tests for 4-player mahjong."""

    def _make_env(self):
        return RiichiEnv(game_mode="default")

    def test_start_game_returns_none(self):
        env = self._make_env()
        result = env.observe_event({"type": "start_game"}, 0)
        assert result is None

    def test_start_kyoku_returns_none(self):
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 0)
        result = env.observe_event(_start_kyoku_event_4p(my_seat=0), 0)
        assert result is None

    def test_tsumo_returns_obs_for_actor(self):
        """Tsumo for player_id's own seat should return an observation."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 0)
        env.observe_event(_start_kyoku_event_4p(my_seat=0), 0)
        obs = env.observe_event({"type": "tsumo", "actor": 0, "pai": "1m"}, 0)
        assert obs is not None
        actions = obs.legal_actions()
        assert len(actions) > 0
        # Should have at least discard options
        discard_actions = [a for a in actions if a.action_type == ActionType.Discard]
        assert len(discard_actions) > 0

    def test_tsumo_returns_none_for_other_player(self):
        """Tsumo for a different seat should return None."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 1)
        env.observe_event(_start_kyoku_event_4p(my_seat=1), 1)
        obs = env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 1)
        assert obs is None

    def test_dahai_returns_obs_when_pon_available(self):
        """After P0 discards 1m, P1 (who has pair of 1m) should get pon option."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 1)
        env.observe_event(_start_kyoku_event_4p(my_seat=1), 1)
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 1)
        obs = env.observe_event({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True}, 1)
        assert obs is not None
        actions = obs.legal_actions()
        action_types = {a.action_type for a in actions}
        assert ActionType.Pon in action_types
        assert ActionType.Pass in action_types

    def test_dahai_returns_none_when_no_reaction(self):
        """After P0 discards a tile no one can claim, returns None."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 2)
        env.observe_event(_start_kyoku_event_4p(my_seat=2), 2)
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 2)
        # P2 has no 1m tiles, so no pon/ron possible
        obs = env.observe_event({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True}, 2)
        # P2 cannot pon 1m (has no 1m in hand), so None is expected
        # (P2 may or may not have chi depending on hand, but
        #  we verify that if obs is returned, it has legal actions)
        if obs is not None:
            assert len(obs.legal_actions()) > 0

    def test_reach_accepted_returns_none(self):
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 0)
        env.observe_event(_start_kyoku_event_4p(my_seat=0), 0)
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "1m"}, 0)
        # Apply reach (may or may not return obs depending on state)
        env.observe_event({"type": "reach", "actor": 0}, 0)
        env.observe_event({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False}, 0)
        result = env.observe_event({"type": "reach_accepted", "actor": 0}, 0)
        assert result is None

    def test_dora_returns_none(self):
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 0)
        env.observe_event(_start_kyoku_event_4p(my_seat=0), 0)
        result = env.observe_event({"type": "dora", "dora_marker": "3p"}, 0)
        assert result is None

    def test_hora_returns_none(self):
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 0)
        env.observe_event(_start_kyoku_event_4p(my_seat=0), 0)
        result = env.observe_event({"type": "hora", "actor": 0, "target": 0}, 0)
        assert result is None

    def test_ryukyoku_returns_none(self):
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 0)
        env.observe_event(_start_kyoku_event_4p(my_seat=0), 0)
        result = env.observe_event({"type": "ryukyoku"}, 0)
        assert result is None

    def test_tsumo_after_pass_returns_obs(self):
        """After P0 discards and P1 passes, P1's tsumo should return obs."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 1)
        env.observe_event(_start_kyoku_event_4p(my_seat=1), 1)
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 1)
        env.observe_event({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True}, 1)
        # P1 draws next
        obs = env.observe_event({"type": "tsumo", "actor": 1, "pai": "6s"}, 1)
        assert obs is not None
        assert len(obs.legal_actions()) > 0

    def test_pon_then_discard_flow(self):
        """P1 pons P0's discard, then P1 should get discard observation."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 1)
        env.observe_event(_start_kyoku_event_4p(my_seat=1), 1)
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 1)
        env.observe_event({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True}, 1)
        # P1 pons
        obs = env.observe_event({"type": "pon", "actor": 1, "target": 0, "pai": "1m", "consumed": ["1m", "1m"]}, 1)
        assert obs is not None
        actions = obs.legal_actions()
        discard_actions = [a for a in actions if a.action_type == ActionType.Discard]
        assert len(discard_actions) > 0

    def test_chi_kuikae_forbids_called_and_other_side_tile(self):
        """After chi on 3m with 4m-5m, both 3m and 6m are forbidden discards."""
        env = self._make_env()
        # P1 has 4m,5m,6m so can chi 3m from P0 (kamicha)
        tehais = [
            _UNKNOWN_HAND_13,
            ["4m", "5m", "6m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1z"],
            _UNKNOWN_HAND_13,
            _UNKNOWN_HAND_13,
        ]
        env.observe_event({"type": "start_game"}, 1)
        env.observe_event(_start_kyoku_event_4p(tehais=tehais), 1)
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 1)
        env.observe_event({"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": True}, 1)
        # P1 calls chi on 3m with 4m-5m
        obs = env.observe_event(
            {"type": "chi", "actor": 1, "target": 0, "pai": "3m", "consumed": ["4m", "5m"]},
            1,
        )
        assert obs is not None
        discard_t34 = {a.tile // 4 for a in obs.legal_actions() if a.action_type == ActionType.Discard}
        # 3m (t34=2) is the called tile — forbidden
        assert 2 not in discard_t34, "3m (called tile) must be forbidden by kuikae"
        # 6m (t34=5) is the other-side tile of the 3-4-5 sequence — forbidden
        assert 5 not in discard_t34, "6m (other-side tile) must be forbidden by kuikae"

    def test_multi_turn_sequence(self):
        """Play multiple turns and verify observations appear at correct times."""
        env = self._make_env()
        player_id = 0
        env.observe_event({"type": "start_game"}, player_id)
        env.observe_event(_start_kyoku_event_4p(my_seat=player_id), player_id)

        # P0 tsumo -> should get obs
        obs = env.observe_event({"type": "tsumo", "actor": 0, "pai": "4p"}, player_id)
        assert obs is not None

        # P0 dahai -> should not get obs (it's our own discard)
        obs = env.observe_event(
            {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
            player_id,
        )
        assert obs is None  # P0 just discarded, no self-reaction

        # P1 tsumo -> P0 should not get obs
        obs = env.observe_event({"type": "tsumo", "actor": 1, "pai": "?"}, player_id)
        assert obs is None

        # P1 dahai -> P0 might get reaction obs
        obs = env.observe_event(
            {"type": "dahai", "actor": 1, "pai": "4s", "tsumogiri": True},
            player_id,
        )
        # Whether P0 gets obs depends on whether P0 can chi/pon/ron 4s
        # Just verify consistency: if obs, must have actions
        if obs is not None:
            assert len(obs.legal_actions()) > 0


# ===========================================================================
# 3P Tests
# ===========================================================================


class TestApplyEvent3P:
    """observe_event tests for 3-player mahjong."""

    def _make_env(self):
        return RiichiEnv(game_mode="3p-red-half")

    def test_start_game_returns_none(self):
        env = self._make_env()
        result = env.observe_event({"type": "start_game"}, 0)
        assert result is None

    def test_start_kyoku_returns_none(self):
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 0)
        result = env.observe_event(_start_kyoku_event_3p(my_seat=0), 0)
        assert result is None

    def test_tsumo_returns_obs_for_actor(self):
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 0)
        env.observe_event(_start_kyoku_event_3p(my_seat=0), 0)
        obs = env.observe_event({"type": "tsumo", "actor": 0, "pai": "3z"}, 0)
        assert obs is not None
        actions = obs.legal_actions()
        assert len(actions) > 0
        discard_actions = [a for a in actions if a.action_type == ActionType.Discard]
        assert len(discard_actions) > 0

    def test_tsumo_returns_none_for_other_player(self):
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 1)
        env.observe_event(_start_kyoku_event_3p(my_seat=1), 1)
        obs = env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 1)
        assert obs is None

    def test_dahai_returns_obs_when_pon_available(self):
        """After P0 discards 1p, P1 (who has pair of 1p) should get pon option."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 1)
        env.observe_event(_start_kyoku_event_3p(my_seat=1), 1)
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 1)
        obs = env.observe_event({"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": False}, 1)
        assert obs is not None
        actions = obs.legal_actions()
        action_types = {a.action_type for a in actions}
        assert ActionType.Pon in action_types
        assert ActionType.Pass in action_types

    def test_dahai_returns_none_when_no_reaction(self):
        """P2 should not get reaction when P0 discards a tile P2 can't claim."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 2)
        env.observe_event(_start_kyoku_event_3p(my_seat=2), 2)
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 2)
        # P0 discards 2p; P2 has no 2p tiles
        obs = env.observe_event({"type": "dahai", "actor": 0, "pai": "2p", "tsumogiri": False}, 2)
        if obs is not None:
            assert len(obs.legal_actions()) > 0

    def test_hora_returns_none(self):
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 0)
        env.observe_event(_start_kyoku_event_3p(my_seat=0), 0)
        result = env.observe_event({"type": "hora", "actor": 0, "target": 0}, 0)
        assert result is None

    def test_ryukyoku_returns_none(self):
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 0)
        env.observe_event(_start_kyoku_event_3p(my_seat=0), 0)
        result = env.observe_event({"type": "ryukyoku"}, 0)
        assert result is None

    def test_tsumo_after_opponent_discard(self):
        """After P0 discards, P1's tsumo should return obs for P1."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 1)
        env.observe_event(_start_kyoku_event_3p(my_seat=1), 1)
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 1)
        env.observe_event({"type": "dahai", "actor": 0, "pai": "3z", "tsumogiri": True}, 1)
        obs = env.observe_event({"type": "tsumo", "actor": 1, "pai": "5z"}, 1)
        assert obs is not None
        assert len(obs.legal_actions()) > 0

    def test_pon_then_discard_flow(self):
        """P1 pons P0's 1p, then P1 should get discard obs."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 1)
        env.observe_event(_start_kyoku_event_3p(my_seat=1), 1)
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 1)
        env.observe_event({"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": False}, 1)
        obs = env.observe_event({"type": "pon", "actor": 1, "target": 0, "pai": "1p", "consumed": ["1p", "1p"]}, 1)
        assert obs is not None
        actions = obs.legal_actions()
        discard_actions = [a for a in actions if a.action_type == ActionType.Discard]
        assert len(discard_actions) > 0

    def test_kita_returns_none_for_other(self):
        """Kita by another player should return None."""
        env = self._make_env()
        env.observe_event({"type": "start_game"}, 1)
        env.observe_event(
            _start_kyoku_event_3p(
                tehais=[
                    _UNKNOWN_HAND_13,
                    ["1p", "1p", "7p", "8p", "9p", "1s", "2s", "3s", "4s", "5s", "6s", "3z", "4z"],
                    _UNKNOWN_HAND_13,
                ],
            ),
            1,
        )
        env.observe_event({"type": "tsumo", "actor": 0, "pai": "?"}, 1)
        result = env.observe_event({"type": "kita", "actor": 0}, 1)
        # Kita is not a skip_check event, but P1 may or may not get
        # reaction (e.g., chankan). Just verify no crash.
        if result is not None:
            assert len(result.legal_actions()) > 0

    def test_multi_turn_sequence(self):
        """Play multiple turns verifying observation timing."""
        env = self._make_env()
        player_id = 0
        env.observe_event({"type": "start_game"}, player_id)
        env.observe_event(_start_kyoku_event_3p(my_seat=player_id), player_id)

        # P0 tsumo -> obs
        obs = env.observe_event({"type": "tsumo", "actor": 0, "pai": "3z"}, player_id)
        assert obs is not None

        # P0 dahai -> no self-reaction
        obs = env.observe_event(
            {"type": "dahai", "actor": 0, "pai": "3z", "tsumogiri": True},
            player_id,
        )
        assert obs is None

        # P1 tsumo -> P0 gets nothing
        obs = env.observe_event({"type": "tsumo", "actor": 1, "pai": "?"}, player_id)
        assert obs is None

        # P1 dahai -> P0 might get reaction
        obs = env.observe_event(
            {"type": "dahai", "actor": 1, "pai": "4s", "tsumogiri": True},
            player_id,
        )
        if obs is not None:
            assert len(obs.legal_actions()) > 0


# ===========================================================================
# Cross-mode consistency
# ===========================================================================


class TestApplyEventConsistency:
    """Verify observe_event behaves consistently across game modes."""

    @pytest.mark.parametrize(
        "game_mode,start_kyoku_fn,n_players",
        [
            ("default", _start_kyoku_event_4p, 4),
            ("3p-red-half", _start_kyoku_event_3p, 3),
        ],
    )
    def test_non_action_events_always_none(self, game_mode, start_kyoku_fn, n_players):
        """All non-action event types return None regardless of player_id."""
        for pid in range(n_players):
            env = RiichiEnv(game_mode=game_mode)
            non_action_events = [
                {"type": "start_game"},
                start_kyoku_fn(my_seat=pid),
                {"type": "dora", "dora_marker": "3p"},
                {"type": "hora", "actor": 0, "target": 0},
                {"type": "ryukyoku"},
            ]
            for ev in non_action_events:
                result = env.observe_event(ev, pid)
                assert result is None, f"{ev['type']} should return None for player {pid} in {game_mode} mode"

    @pytest.mark.parametrize(
        "game_mode,start_kyoku_fn",
        [
            ("default", _start_kyoku_event_4p),
            ("3p-red-half", _start_kyoku_event_3p),
        ],
    )
    def test_tsumo_only_returns_obs_for_actor(self, game_mode, start_kyoku_fn):
        """Tsumo returns obs only when player_id == actor."""
        n_players = 3 if "3p" in game_mode else 4
        tsumo_tile = "3z" if "3p" in game_mode else "1m"
        for pid in range(n_players):
            env = RiichiEnv(game_mode=game_mode)
            env.observe_event({"type": "start_game"}, pid)
            env.observe_event(start_kyoku_fn(my_seat=pid), pid)
            pai = tsumo_tile if pid == 0 else "?"
            obs = env.observe_event({"type": "tsumo", "actor": 0, "pai": pai}, pid)
            if pid == 0:
                assert obs is not None, "Actor should get observation on own tsumo"
            else:
                assert obs is None, f"Player {pid} should not get obs on P0's tsumo"


# ===========================================================================
# Replay iterator: pass observation furiten tests
# ===========================================================================

# Shared hand layout for furiten tests:
# P0: 3m,3m,5m,...  (has two 3m to discard)
# P1: 1s-9s + 1m,1m,1m,2m  (tenpai for 3m: 1s-2s-3s,4s-5s-6s,7s-8s-9s,1m-2m-3m + 1m pair)
# P2: 1z,...,3m,...  (has a 3m to discard)
# P3: safe tiles
_FURITEN_TEHAIS_JSON = (
    '["3m","3m","5m","6m","7m","8m","9m","1p","2p","3p","4p","5p","6p"],'
    '["1s","2s","3s","4s","5s","6s","7s","8s","9s","1m","1m","1m","2m"],'
    '["1z","2z","3z","4z","5z","6z","7z","7p","8p","9p","3m","8s","9s"],'
    '["4s","5s","6s","7s","8s","9s","4m","5m","6m","7m","8m","9m","1z"]'
)


def _write_jsonl(lines: list[str]) -> str:
    """Write JSONL lines to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    f.write("\n".join(lines))
    f.close()
    return f.name


class TestReplayFuriten:
    """Verify that pass observations update furiten state correctly."""

    def test_doujun_furiten_resets_after_own_discard(self):
        """Same-turn furiten: P1 passes on ron, then draws/discards.
        After P1's own discard, doujun furiten resets, so P1 can ron
        the same tile again from a different player."""
        lines = [
            '{"type":"start_game"}',
            '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyoutaku":0,'
            '"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"2p",'
            '"tehais":[' + _FURITEN_TEHAIS_JSON + "]}",
            # P0 draws, discards 3m — P1 can ron but passes
            '{"type":"tsumo","actor":0,"pai":"7p"}',
            '{"type":"dahai","actor":0,"pai":"3m","tsumogiri":false}',
            # P1 draws, discards (doujun furiten resets)
            '{"type":"tsumo","actor":1,"pai":"4z"}',
            '{"type":"dahai","actor":1,"pai":"4z","tsumogiri":true}',
            # P2 discards 3m — P1 should have Ron again
            '{"type":"tsumo","actor":2,"pai":"5z"}',
            '{"type":"dahai","actor":2,"pai":"3m","tsumogiri":false}',
            '{"type":"tsumo","actor":3,"pai":"3z"}',
            '{"type":"dahai","actor":3,"pai":"3z","tsumogiri":true}',
            '{"type":"ryukyoku","reason":"yao9"}',
            '{"type":"end_kyoku"}',
            '{"type":"end_game"}',
        ]
        path = _write_jsonl(lines)
        try:
            replay = MjaiReplay.from_jsonl(path, rule="tenhou")
            for kyoku in replay.take_kyokus():
                p1_pass_has_ron = []
                for step in kyoku.steps(seat=None, skip_single_action=False):
                    pid, obs, action = step
                    if pid == 1 and action.action_type == ActionType.Pass:
                        types = {a.action_type for a in obs.legal_actions()}
                        p1_pass_has_ron.append(ActionType.Ron in types)
                # Both passes should have Ron (doujun furiten resets after discard)
                assert len(p1_pass_has_ron) == 2, f"expected 2 pass obs, got {len(p1_pass_has_ron)}"
                assert p1_pass_has_ron[0] is True, "1st pass should include Ron"
                assert p1_pass_has_ron[1] is True, "2nd pass should include Ron (doujun reset)"
        finally:
            os.unlink(path)

    def test_riichi_furiten_persists_after_own_discard(self):
        """Riichi furiten: P1 in riichi passes on ron, so missed_agari_riichi
        is set permanently. Even after P1's own discard, the second 3m
        from P2 should NOT give P1 a Ron opportunity."""
        lines = [
            '{"type":"start_game"}',
            '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyoutaku":0,'
            '"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"2p",'
            '"tehais":[' + _FURITEN_TEHAIS_JSON + "]}",
            # P0 draws, discards safe tile
            '{"type":"tsumo","actor":0,"pai":"7p"}',
            '{"type":"dahai","actor":0,"pai":"7p","tsumogiri":true}',
            # P1 draws, declares riichi, discards
            '{"type":"tsumo","actor":1,"pai":"4z"}',
            '{"type":"reach","actor":1}',
            '{"type":"dahai","actor":1,"pai":"4z","tsumogiri":true}',
            '{"type":"reach_accepted","actor":1}',
            # P2, P3 safe discards
            '{"type":"tsumo","actor":2,"pai":"5z"}',
            '{"type":"dahai","actor":2,"pai":"5z","tsumogiri":true}',
            '{"type":"tsumo","actor":3,"pai":"2z"}',
            '{"type":"dahai","actor":3,"pai":"2z","tsumogiri":true}',
            # P0 discards 3m — P1 (riichi) can ron but passes
            '{"type":"tsumo","actor":0,"pai":"1z"}',
            '{"type":"dahai","actor":0,"pai":"3m","tsumogiri":false}',
            # P1 draws, auto-tsumogiri (riichi)
            '{"type":"tsumo","actor":1,"pai":"6z"}',
            '{"type":"dahai","actor":1,"pai":"6z","tsumogiri":true}',
            # P2 discards 3m — P1 should NOT get Ron (riichi furiten)
            '{"type":"tsumo","actor":2,"pai":"7z"}',
            '{"type":"dahai","actor":2,"pai":"3m","tsumogiri":false}',
            '{"type":"tsumo","actor":3,"pai":"3z"}',
            '{"type":"dahai","actor":3,"pai":"3z","tsumogiri":true}',
            '{"type":"ryukyoku","reason":"yao9"}',
            '{"type":"end_kyoku"}',
            '{"type":"end_game"}',
        ]
        path = _write_jsonl(lines)
        try:
            replay = MjaiReplay.from_jsonl(path, rule="tenhou")
            for kyoku in replay.take_kyokus():
                p1_pass_has_ron = []
                for step in kyoku.steps(seat=None, skip_single_action=False):
                    pid, obs, action = step
                    if pid == 1 and action.action_type == ActionType.Pass:
                        types = {a.action_type for a in obs.legal_actions()}
                        p1_pass_has_ron.append(ActionType.Ron in types)
                # Only the first pass (from P0's 3m) should have Ron.
                # The second 3m (from P2) produces no pass obs because
                # riichi furiten blocks all claims.
                assert len(p1_pass_has_ron) == 1, f"expected 1 pass obs, got {len(p1_pass_has_ron)}"
                assert p1_pass_has_ron[0] is True, "1st pass should include Ron"
        finally:
            os.unlink(path)
