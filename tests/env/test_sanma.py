"""Tests for 3-player (sanma) mahjong via RiichiEnv."""

import math
import struct

import pytest

from riichienv import (
    Action,
    ActionType,
    GameType,
    Phase,
    RiichiEnv,
    calculate_score,
)
from riichienv._riichienv import Observation3P
from riichienv.consts import N_TILE_TYPES_3P
from riichienv.convert import tid_to_mjai

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_sanma_env(seed=42, game_mode=GameType.SAN_HANCHAN, **kwargs):
    """Create a 3P environment and reset it."""
    env = RiichiEnv(game_mode=game_mode, seed=seed, **kwargs)
    obs = env.reset()
    return env, obs


def _play_one_turn(env, obs):
    """Discard the last tile in the current player's hand, handle WaitResponse."""
    pid = env.current_player
    o = obs[pid]
    tile = o.hand[-1]
    obs = env.step({pid: Action(ActionType.DISCARD, tile=tile)})
    while env.phase == Phase.WaitResponse:
        actions = {p: Action(ActionType.PASS) for p in env.active_players}
        obs = env.step(actions)
    return obs


# ===========================================================================
# Initialization
# ===========================================================================


class TestSanmaInitialization:
    def test_num_players(self):
        env, _ = _create_sanma_env()
        assert env.num_players == 3

    def test_starting_scores(self):
        env, _ = _create_sanma_env()
        assert env.scores() == [35000, 35000, 35000]

    def test_hand_counts(self):
        """Dealer has 14 tiles, others have 13."""
        env, _ = _create_sanma_env()
        assert len(env.hands) == 3
        assert len(env.hands[0]) == 14  # dealer
        assert len(env.hands[1]) == 13
        assert len(env.hands[2]) == 13

    def test_wall_size(self):
        """108 total - (14+13+13) initial deal = 68 remaining."""
        env, _ = _create_sanma_env()
        assert len(env.wall) == 68

    def test_observation_type(self):
        """3P games should return Observation3P, not Observation."""
        _, obs = _create_sanma_env()
        assert isinstance(obs[0], Observation3P)

    def test_initial_phase(self):
        env, obs = _create_sanma_env()
        assert env.phase == Phase.WaitAct
        assert env.current_player == 0  # dealer starts
        assert list(obs.keys()) == [0]

    def test_mjai_log_start(self):
        env, _ = _create_sanma_env()
        assert env.mjai_log[0]["type"] == "start_game"
        assert env.mjai_log[1]["type"] == "start_kyoku"
        assert env.mjai_log[2]["type"] == "tsumo"

    def test_tiles_exclude_manzu_2_to_8(self):
        """In sanma, tiles 2m-8m (tile types 1-7, i.e. tile IDs 4-31) are excluded."""
        env, _ = _create_sanma_env()
        all_tiles = set()
        for h in env.hands:
            all_tiles.update(h)
        all_tiles.update(env.wall)
        # Dora indicators are inside the wall conceptually
        all_tiles.update(env.dora_indicators)
        for tid in all_tiles:
            tile_type = tid // 4
            assert tile_type < 1 or tile_type > 7, (
                f"Tile {tid} (type {tile_type}) is manzu 2-8, should be excluded in sanma"
            )

    @pytest.mark.parametrize(
        "game_mode",
        [GameType.SAN_IKKYOKU, GameType.SAN_TONPUSEN, GameType.SAN_HANCHAN],
    )
    def test_all_sanma_game_modes(self, game_mode):
        env = RiichiEnv(game_mode=game_mode, seed=42)
        obs = env.reset()
        assert env.num_players == 3
        assert isinstance(obs[0], Observation3P)


# ===========================================================================
# Player Rotation
# ===========================================================================


class TestSanmaRotation:
    def test_player_rotation_3_players(self):
        """Turns rotate 0 -> 1 -> 2 -> 0."""
        env, obs = _create_sanma_env()
        expected_order = [0, 1, 2, 0]
        for expected_pid in expected_order:
            assert env.current_player == expected_pid
            obs = _play_one_turn(env, obs)


# ===========================================================================
# No Chi in Sanma
# ===========================================================================


class TestSanmaNoChi:
    def test_no_chi_in_legal_actions(self):
        """Chi is never available in 3-player mahjong."""
        env, obs = _create_sanma_env()
        # Play several rounds and check all legal actions
        for _ in range(6):
            if env.is_done:
                break
            pid = env.current_player
            o = obs[pid]
            for a in o.legal_actions():
                assert a.action_type != ActionType.CHI, "Chi must not appear in sanma"
            obs = _play_one_turn(env, obs)

    def test_shimocha_cannot_chi(self):
        """Even when the next player has sequential tiles, Chi is impossible."""
        env, _ = _create_sanma_env()
        # Give P0 a tile to discard, P1 has sequential tiles
        h = env.hands
        # 1m=0, 9m=32..35, 1p=36..39, 2p=40..43, 3p=44..47
        h[0] = [36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88]
        h[1] = [37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85]
        h[1].sort()
        env.hands = h
        env.current_player = 0
        env.active_players = [0]
        env.drawn_tile = 88

        # P0 discards 1p (36) - P1 has 2p(41),3p(45) but cannot Chi
        obs = env.step({0: Action(ActionType.DISCARD, tile=36)})
        if env.phase == Phase.WaitResponse:
            for pid in env.active_players:
                if pid in obs:
                    for a in obs[pid].legal_actions():
                        assert a.action_type != ActionType.CHI


# ===========================================================================
# Pon in Sanma
# ===========================================================================


class TestSanmaPon:
    def test_pon_claim(self):
        """Pon should work in 3P games."""
        env, _ = _create_sanma_env()
        h = env.hands
        # P0 has 1p (36) to discard
        h[0] = [36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88]
        # P1 has pair of 1p (37, 38) for pon
        h[1] = [37, 38, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89]
        h[1].sort()
        env.hands = h
        env.current_player = 0
        env.active_players = [0]
        env.drawn_tile = 88

        obs = env.step({0: Action(ActionType.DISCARD, tile=36)})
        assert env.phase == Phase.WaitResponse

        # P1 should have pon option
        obs_p1 = obs[1]
        pon_actions = [a for a in obs_p1.legal_actions() if a.action_type == ActionType.PON]
        assert len(pon_actions) > 0

        # Execute pon
        env.step({1: pon_actions[0]})
        assert env.current_player == 1
        assert env.phase == Phase.WaitAct
        assert len(env.melds[1]) == 1


# ===========================================================================
# Observation3P Encoding
# ===========================================================================


class TestSanmaObservation:
    def test_observation_fields(self):
        _, obs = _create_sanma_env()
        o = obs[0]
        assert o.player_id == 0
        assert len(o.hands) == 3
        assert len(o.melds) == 3
        assert len(o.discards) == 3
        assert len(o.scores) == 3
        assert len(o.riichi_declared) == 3

    def test_action_space_size(self):
        _, obs = _create_sanma_env()
        # 3P action space is smaller than 4P (60 vs 181)
        assert obs[0].action_space_size == 60

    def test_encode_shape(self):
        """encode() returns 74 channels * 27 tiles * 4 bytes."""
        _, obs = _create_sanma_env()
        enc = obs[0].encode()
        assert len(enc) == 74 * N_TILE_TYPES_3P * 4

    def test_encode_extended_shape(self):
        """encode_extended() returns 215 channels * 27 tiles * 4 bytes."""
        _, obs = _create_sanma_env()
        enc = obs[0].encode_extended()
        assert len(enc) == 215 * N_TILE_TYPES_3P * 4

    def test_mask_size(self):
        """mask() length should match action_space_size."""
        _, obs = _create_sanma_env()
        o = obs[0]
        mask = o.mask()
        assert len(mask) == o.action_space_size

    def test_mask_has_legal_actions(self):
        """At least one action should be legal in the mask."""
        _, obs = _create_sanma_env()
        mask = obs[0].mask()
        assert sum(mask) > 0

    def test_find_action(self):
        _, obs = _create_sanma_env()
        o = obs[0]
        mask = o.mask()
        # Find first legal action id
        for aid, val in enumerate(mask):
            if val == 1:
                action = o.find_action(aid)
                assert action is not None
                break

    def test_encode_discard_history_decay(self):
        env, obs = _create_sanma_env()
        # Play a few turns to populate discards
        for _ in range(3):
            if env.is_done:
                break
            obs = _play_one_turn(env, obs)
        if not env.is_done:
            o = obs[env.current_player]
            enc = o.encode_discard_history_decay()
            # (3, 27) array of f32 = 3 * 27 * 4 bytes
            assert len(enc) == 3 * N_TILE_TYPES_3P * 4

    def test_encode_shanten_efficiency(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_shanten_efficiency()
        # (3, 4) array of f32 = 3 * 4 * 4 bytes
        assert len(enc) == 3 * 4 * 4

    def test_encode_yaku_possibility(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_yaku_possibility()
        # (3, 21, 2) array of f32
        assert len(enc) == 3 * 21 * 2 * 4

    def test_encode_kawa_overview(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_kawa_overview()
        # (3, 7, 27) array of f32
        assert len(enc) == 3 * 7 * N_TILE_TYPES_3P * 4

    def test_encode_fuuro_overview(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_fuuro_overview()
        # (3, 4, 5, 27) array of f32
        assert len(enc) == 3 * 4 * 5 * N_TILE_TYPES_3P * 4

    def test_encode_ankan_overview(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_ankan_overview()
        # (3, 27) array of f32
        assert len(enc) == 3 * N_TILE_TYPES_3P * 4

    def test_encode_action_availability(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_action_availability()
        # (11,) array of f32
        assert len(enc) == 11 * 4

    def test_encode_riichi_sutehais(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_riichi_sutehais()
        # (2, 3) array of f32 for opponents
        assert len(enc) == 2 * 3 * 4

    def test_encode_last_tedashis(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_last_tedashis()
        # (2, 3) array of f32
        assert len(enc) == 2 * 3 * 4

    def test_encode_pass_context(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_pass_context()
        # (3,) array of f32
        assert len(enc) == 3 * 4

    def test_encode_discard_candidates(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_discard_candidates()
        # (5,) array of f32
        assert len(enc) == 5 * 4

    def test_to_dict(self):
        _, obs = _create_sanma_env()
        d = obs[0].to_dict()
        assert d["player_id"] == 0
        assert len(d["hands"]) == 3
        assert len(d["scores"]) == 3


# ===========================================================================
# Serialization
# ===========================================================================


class TestSanmaSerialization:
    def test_round_trip(self):
        _, obs = _create_sanma_env()
        o = obs[0]
        b64 = o.serialize_to_base64()
        restored = Observation3P.deserialize_from_base64(b64)
        assert restored.player_id == o.player_id
        assert restored.hands == o.hands
        assert restored.scores == o.scores
        assert restored.dora_indicators == o.dora_indicators
        assert restored.riichi_declared == o.riichi_declared
        assert restored.honba == o.honba
        assert restored.round_wind == o.round_wind
        assert restored.oya == o.oya

    def test_legal_actions_preserved(self):
        _, obs = _create_sanma_env()
        o = obs[0]
        restored = Observation3P.deserialize_from_base64(o.serialize_to_base64())
        orig = o.legal_actions()
        rest = restored.legal_actions()
        assert len(rest) == len(orig)
        for a, b in zip(orig, rest):
            assert a.action_type == b.action_type
            assert a.tile == b.tile

    @pytest.mark.parametrize("seed", [0, 1, 99, 12345])
    def test_round_trip_multiple_seeds(self, seed):
        _, obs = _create_sanma_env(seed=seed)
        o = obs[0]
        restored = Observation3P.deserialize_from_base64(o.serialize_to_base64())
        assert restored.player_id == o.player_id
        assert restored.hands == o.hands


# ===========================================================================
# Scoring
# ===========================================================================


class TestSanmaScoring:
    def test_points_basic(self):
        env, _ = _create_sanma_env()
        pts = env.points("basic")
        assert len(pts) == 3
        assert pts == [40.0, 0.0, -40.0]

    def test_ranks(self):
        env, _ = _create_sanma_env()
        ranks = env.ranks()
        assert len(ranks) == 3
        # All scores equal -> ranks based on seat
        assert set(ranks) == {1, 2, 3}

    def test_tsumo_deltas_3_players(self):
        """Tsumo in 3P: winner gains, two others lose; deltas sum to 0."""
        env, _ = _create_sanma_env()
        # Set up a tsumo win for P0
        # 1p*3(36,37,38), 2p*3(40,41,42), 3p*3(44,45,46), 9m*3(32,33,34), 1m(0)
        # Wait: 1m pair
        hand = [36, 37, 38, 40, 41, 42, 44, 45, 46, 32, 33, 34, 0]
        h = env.hands
        h[0] = sorted(hand)
        env.hands = h
        env.drawn_tile = 1  # 1m - completes the pair
        env.current_player = 0
        env.is_first_turn = False
        d = env.discards
        d[0].append(100)  # avoid tenhou check
        env.discards = d

        actions = env._get_legal_actions(0)
        tsumo = next((a for a in actions if a.action_type == ActionType.TSUMO), None)
        assert tsumo is not None, f"Expected tsumo in legal actions: {actions}"

        env.step({0: Action(ActionType.TSUMO)})
        hora = next(e for e in reversed(env.mjai_log) if e["type"] == "hora")
        assert hora["tsumo"] is True
        deltas = hora["deltas"]
        assert len(deltas) == 3
        assert deltas[0] > 0
        assert deltas[1] < 0
        assert deltas[2] < 0
        assert sum(deltas) == 0

    def test_ron_deltas_3_players(self):
        """Ron in 3P: winner gains, discarder loses; deltas sum to 0."""
        env, _ = _create_sanma_env()
        # P1 tenpai: 1p*3, 2p*3, 3p*3, 9m*3, 1m (waiting 1m)
        p1_hand = [36, 37, 38, 40, 41, 42, 44, 45, 46, 32, 33, 34, 0]
        h = env.hands
        h[1] = sorted(p1_hand)
        env.hands = h

        # P0 discards 1m (tile 1)
        h = env.hands
        h[0] = [1] + [48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96]
        h[0].sort()
        env.hands = h
        env.current_player = 0
        env.active_players = [0]
        env.drawn_tile = 96

        obs = env.step({0: Action(ActionType.DISCARD, tile=1)})
        assert env.phase == Phase.WaitResponse
        assert 1 in obs

        ron_acts = [a for a in obs[1].legal_actions() if a.action_type == ActionType.RON]
        assert len(ron_acts) == 1

        env.step({1: ron_acts[0]})
        hora = next(e for e in reversed(env.mjai_log) if e["type"] == "hora")
        deltas = hora["deltas"]
        assert len(deltas) == 3
        assert deltas[1] > 0  # P1 wins
        assert deltas[0] < 0  # P0 pays
        assert deltas[2] == 0  # P2 unaffected
        assert sum(deltas) == 0

    def test_calculate_score_3p(self):
        """calculate_score with num_players=3 should work."""
        score = calculate_score(han=3, fu=30, is_oya=False, is_tsumo=False, honba=0, num_players=3)
        assert score is not None


# ===========================================================================
# Game Flow
# ===========================================================================


class TestSanmaGameFlow:
    def test_play_full_round(self):
        """Play through discards until the round ends (exhaustive draw or win)."""
        env, obs = _create_sanma_env(seed=7)
        turns = 0
        while not env.is_done and turns < 200:
            pid = env.current_player
            o = obs[pid]
            # Try tsumo first if available
            legals = o.legal_actions()
            tsumo = next((a for a in legals if a.action_type == ActionType.TSUMO), None)
            if tsumo:
                obs = env.step({pid: tsumo})
            else:
                tile = o.hand[-1]
                obs = env.step({pid: Action(ActionType.DISCARD, tile=tile)})
            while env.phase == Phase.WaitResponse and not env.is_done:
                actions = {p: Action(ActionType.PASS) for p in env.active_players}
                obs = env.step(actions)
            turns += 1
        assert env.is_done or turns >= 200

    def test_seeded_determinism(self):
        """Same seed produces identical hands."""
        env1, _ = _create_sanma_env(seed=123)
        env2, _ = _create_sanma_env(seed=123)
        assert env1.hands == env2.hands
        assert env1.wall == env2.wall

    def test_different_seeds_differ(self):
        """Different seeds produce different hands."""
        env1, _ = _create_sanma_env(seed=1)
        env2, _ = _create_sanma_env(seed=2)
        assert env1.hands != env2.hands

    def test_mjai_events_visible_to_all(self):
        """All players should see mjai events (with masked hands)."""
        env, obs = _create_sanma_env()
        # P0 discards
        tile = obs[0].hand[-1]
        obs = env.step({0: Action(ActionType.DISCARD, tile=tile)})

        # Handle WaitResponse
        while env.phase == Phase.WaitResponse:
            actions = {p: Action(ActionType.PASS) for p in env.active_players}
            obs = env.step(actions)

        # P1 should see events including start_kyoku with 3 player tehais
        o = obs[env.current_player]
        start_kyoku = o.events[1]
        assert start_kyoku["type"] == "start_kyoku"
        assert len(start_kyoku["tehais"]) == 3

    def test_start_kyoku_scores(self):
        """start_kyoku event should have 3 scores."""
        env, obs = _create_sanma_env()
        start_kyoku = next(e for e in env.mjai_log if e["type"] == "start_kyoku")
        assert len(start_kyoku["scores"]) == 3
        assert start_kyoku["scores"] == [35000, 35000, 35000]


# ===========================================================================
# Select action from MJAI
# ===========================================================================


class TestSanmaMjaiAction:
    def test_select_action_from_mjai_discard(self):
        _, obs = _create_sanma_env()
        o = obs[0]
        tile = o.hand[0]
        mjai_str = tid_to_mjai(tile)
        action = o.select_action_from_mjai({"type": "dahai", "pai": mjai_str, "actor": 0})
        assert action is not None
        assert action.action_type == ActionType.DISCARD

    def test_select_action_from_mjai_pass(self):
        """Pass should always be selectable during WaitResponse."""
        env, _ = _create_sanma_env()
        h = env.hands
        # P0 has 1p to discard, P1 has pair of 1p for pon
        h[0] = [36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88]
        h[1] = [37, 38, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89]
        h[1].sort()
        env.hands = h
        env.current_player = 0
        env.active_players = [0]
        env.drawn_tile = 88

        obs = env.step({0: Action(ActionType.DISCARD, tile=36)})
        if 1 in obs:
            o = obs[1]
            action = o.select_action_from_mjai({"type": "none"})
            assert action is not None
            assert action.action_type == ActionType.PASS


# ===========================================================================
# Encode values sanity
# ===========================================================================


class TestSanmaEncodeSanity:
    def test_encode_values_finite(self):
        """All encoded float values should be finite."""
        _, obs = _create_sanma_env()
        enc = obs[0].encode()
        floats = struct.unpack(f"<{len(enc) // 4}f", enc)
        for v in floats:
            assert not math.isnan(v), "NaN detected in encode output"
            assert abs(v) < 1e10, f"Extreme value detected: {v}"

    def test_encode_extended_values_finite(self):
        _, obs = _create_sanma_env()
        enc = obs[0].encode_extended()
        floats = struct.unpack(f"<{len(enc) // 4}f", enc)
        for v in floats:
            assert not math.isnan(v), "NaN detected in encode_extended output"
            assert abs(v) < 1e10, f"Extreme value detected: {v}"
