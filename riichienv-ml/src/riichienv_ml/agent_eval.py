"""AgentEvaluator — evaluate a hero model against fixed Agent opponents.

Satisfies the :class:`ThirdPartyEvaluator` protocol and is registered as
the ``riichienv`` entry-point in ``riichienv_ml.evaluators``.

Usage via plugin system::

    from riichienv_ml.evaluator import load_evaluator
    ev = load_evaluator("riichienv", ...)
    metrics = ev.evaluate(hero_weights, num_episodes=30)
"""
from __future__ import annotations

import time

import numpy as np
import torch
from loguru import logger

from riichienv_ml.agents import Agent
from riichienv_ml.config import (
    GAME_PARAMS,
    OpponentConfig,
    import_class,
)


class AgentEvaluator:
    """1-vs-N evaluator: hero model vs fixed Agent opponents.

    Args:
        model_class: Dotted path to the hero model class.
        model_config: Dict of model constructor kwargs.
        encoder_class: Dotted path to the observation encoder class.
        device: Device for hero model inference.
        eval_device: Device for opponent model inference.
        opponents: List of ``OpponentConfig`` dicts (config_path + model_path).
        n_players: Number of players (3 or 4).
        tile_dim: Tile dimension matching the game mode.
        episodes_per_seat: Episodes to play per hero seat position.
        **kwargs: Ignored (for compatibility with ``load_evaluator``).
    """

    METRICS_PREFIX = "agent_eval"

    def __init__(
        self,
        *,
        model_class: str,
        model_config: dict,
        encoder_class: str,
        device: str = "cuda",
        eval_device: str = "cpu",
        opponents: list[dict | OpponentConfig] = (),
        n_players: int = 3,
        tile_dim: int | None = None,
        episodes_per_seat: int | None = None,
        **kwargs,
    ):
        self.device = torch.device(device)
        self.eval_device = eval_device
        self.n_players = n_players

        game_params = GAME_PARAMS[n_players]
        self.tile_dim = tile_dim or game_params["tile_dim"]
        self.game_mode = game_params["game_mode"]
        self.starting_scores = game_params["starting_scores"]
        self.episodes_per_seat = episodes_per_seat

        # Build hero model
        ModelClass = import_class(model_class)
        self.hero_model = ModelClass(**model_config).to(self.device)
        self.hero_model.eval()

        # Build hero encoder
        EncoderClass = import_class(encoder_class)
        self.hero_encoder = EncoderClass(tile_dim=self.tile_dim)

        # Build opponent agents (pad to N-1 if needed)
        opp_cfgs = [
            OpponentConfig(**o) if isinstance(o, dict) else o
            for o in opponents
        ]
        n_opponents = n_players - 1
        if len(opp_cfgs) == 0:
            raise ValueError("At least one opponent config is required")
        while len(opp_cfgs) < n_opponents:
            opp_cfgs.append(opp_cfgs[-1])
        opp_cfgs = opp_cfgs[:n_opponents]

        self.opponents: list[Agent] = []
        for oc in opp_cfgs:
            agent = Agent(
                config_path=oc.config_path,
                model_path=oc.model_path,
                device=self.eval_device,
            )
            self.opponents.append(agent)

        logger.info(
            f"AgentEvaluator: {n_players}P, hero={device}, "
            f"{len(self.opponents)} opponent(s) on {eval_device}"
        )

    # ------------------------------------------------------------------
    # ThirdPartyEvaluator protocol
    # ------------------------------------------------------------------

    def evaluate(self, hero_weights: dict, num_episodes: int = 48) -> dict:
        """Run evaluation games and return metrics dict.

        Hero is rotated across all seat positions for fairness:
        ``num_episodes`` games total, distributed evenly across N seats.
        """
        # Load hero weights
        self.hero_model.load_state_dict(hero_weights, strict=False)
        self.hero_model.eval()

        from riichienv import RiichiEnv

        eps_per_seat = self.episodes_per_seat
        if eps_per_seat is None:
            eps_per_seat = max(1, -(-num_episodes // self.n_players))  # ceil division

        all_ranks: list[int] = []
        all_scores: list[float] = []
        t0 = time.time()

        for hero_seat in range(self.n_players):
            for _ in range(eps_per_seat):
                rank, score = self._play_one_game(RiichiEnv, hero_seat)
                all_ranks.append(rank)
                all_scores.append(score)

        elapsed = time.time() - t0
        total_eps = len(all_ranks)
        ranks_arr = np.array(all_ranks, dtype=np.float64)
        scores_arr = np.array(all_scores, dtype=np.float64)

        pfx = self.METRICS_PREFIX
        metrics = {
            f"{pfx}/rank_mean": float(ranks_arr.mean()),
            f"{pfx}/rank_se": float(ranks_arr.std() / np.sqrt(total_eps)),
            f"{pfx}/score_mean": float(scores_arr.mean()),
            f"{pfx}/episodes": total_eps,
            f"{pfx}/time": elapsed,
        }
        # Per-rank rates
        for r in range(1, self.n_players + 1):
            metrics[f"{pfx}/{r}st_rate" if r == 1
                    else f"{pfx}/{r}nd_rate" if r == 2
                    else f"{pfx}/{r}rd_rate" if r == 3
                    else f"{pfx}/{r}th_rate"] = float((ranks_arr == r).mean())

        return metrics

    def metrics_to_logline(self, metrics: dict) -> str:
        pfx = self.METRICS_PREFIX
        parts = [
            f"rank={metrics[f'{pfx}/rank_mean']:.2f}"
            f"\u00b1{metrics[f'{pfx}/rank_se']:.2f}",
            f"score={metrics[f'{pfx}/score_mean']:.0f}",
        ]
        for r in range(1, self.n_players + 1):
            suffix = (
                "1st_rate" if r == 1
                else "2nd_rate" if r == 2
                else "3rd_rate" if r == 3
                else f"{r}th_rate"
            )
            key = f"{pfx}/{suffix}"
            if key in metrics:
                parts.append(f"{suffix}={metrics[key]:.1%}")
        parts.append(f"{metrics[f'{pfx}/episodes']} eps")
        parts.append(f"{metrics[f'{pfx}/time']:.1f}s")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _play_one_game(self, RiichiEnv, hero_seat: int) -> tuple[int, float]:
        """Play a single game and return (hero_rank, hero_score)."""
        env = RiichiEnv(game_mode=self.game_mode)
        obs_dict = env.reset(starting_scores=self.starting_scores)

        # Map seat -> agent (opponents fill non-hero seats)
        opp_idx = 0
        seat_agents: dict[int, Agent] = {}
        for s in range(self.n_players):
            if s != hero_seat:
                seat_agents[s] = self.opponents[opp_idx]
                opp_idx += 1

        while not env.done():
            actions = {}
            for pid, obs in obs_dict.items():
                if pid == hero_seat:
                    actions[pid] = self._hero_act(obs)
                else:
                    actions[pid] = seat_agents[pid].act(obs)
            obs_dict = env.step(actions)

        ranks = env.ranks()   # 1-indexed
        scores = env.scores()
        return ranks[hero_seat], scores[hero_seat]

    def _hero_act(self, obs):
        """Select an action for the hero using the hero model."""
        feat = self.hero_encoder.encode(obs)
        mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()

        feat_batch = feat.to(self.device).unsqueeze(0)
        mask_t = torch.from_numpy(mask).to(self.device).unsqueeze(0)

        output = self.hero_model(feat_batch)
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits.masked_fill(mask_t == 0, -1e9)
        action_idx = logits.argmax(dim=1).item()

        action = obs.find_action(action_idx)
        if action is not None:
            return action

        legals = obs.legal_actions()
        if legals:
            return legals[0]
        raise ValueError(f"No legal action for action_id={action_idx}")
