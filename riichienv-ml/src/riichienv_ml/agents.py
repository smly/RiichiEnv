"""Agents for RiichiEnv loaded from YAML training configs.

Usage::

    from riichienv import RiichiEnv
    from riichienv_ml.agents import Agent

    agents = {
        0: Agent("configs/4p/ppo.yml"),
        1: Agent("configs/4p/bc_model.yml", device="cuda"),
    }

    env = RiichiEnv(game_mode="4p-red-half")
    obs_dict = env.reset()
    while not env.done():
        actions = {}
        for pid, obs in obs_dict.items():
            actions[pid] = agents[pid].act(obs)
        obs_dict = env.step(actions)
"""

import warnings

import numpy as np
import torch
import yaml

from riichienv_ml.config import load_config, import_class


class Agent:
    """Agent that plays in RiichiEnv using a model from a training config.

    Loads model architecture, encoder, and weights from a YAML config file.
    Auto-detects the config section (ppo / bc / cql).

    Args:
        config_path: Path to a training config YAML.
        model_path: Checkpoint path override.  When ``None``, inferred from
            config (``load_model`` for ppo, ``output`` for bc/cql).
        device: Torch device for model inference.
    """

    _SECTIONS = ("ppo", "bc", "cql")

    def __init__(
        self,
        config_path: str,
        model_path: str | None = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device)

        # Detect config section and load
        section, sub_cfg = self._load_section(config_path)
        game = sub_cfg.game

        # Resolve model path
        if model_path is None:
            model_path = self._find_model_path(sub_cfg, section)

        # Build model from config
        ModelClass = import_class(sub_cfg.model_class)
        self.model = ModelClass(**sub_cfg.model.model_dump()).to(self.device)

        # Load weights
        state = torch.load(
            model_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._load_weights(state)
        self.model.eval()

        # Build encoder
        EncoderClass = import_class(sub_cfg.encoder_class)
        self.encoder = EncoderClass(tile_dim=game.tile_dim)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @classmethod
    def _load_section(cls, config_path: str):
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Config must be a YAML mapping with one of "
                f"{'/'.join(cls._SECTIONS)}, got {type(raw).__name__}")
        for key in cls._SECTIONS:
            if key in raw:
                cfg = load_config(config_path)
                return key, getattr(cfg, key)
        raise ValueError(
            f"No recognized section ({'/'.join(cls._SECTIONS)}) in {config_path}")

    @staticmethod
    def _find_model_path(sub_cfg, section: str) -> str:
        if section == "ppo" and getattr(sub_cfg, "load_model", None):
            return sub_cfg.load_model
        if getattr(sub_cfg, "output", None):
            return sub_cfg.output
        raise ValueError(
            "Cannot determine model path from config; "
            "pass model_path explicitly")

    def _load_weights(self, state: dict):
        """Load state dict with QNetwork → ActorCritic auto-conversion."""
        has_actor = any(k.startswith("actor_head.") for k in state)
        has_head = any(k.startswith("head.") for k in state)
        if has_head and not has_actor:
            new_state = {}
            for k, v in state.items():
                if k.startswith("head."):
                    new_state[k.replace("head.", "actor_head.")] = v
                else:
                    new_state[k] = v
            result = self.model.load_state_dict(new_state, strict=False)
        else:
            result = self.model.load_state_dict(state, strict=False)
        if result.missing_keys:
            warnings.warn(
                f"Missing keys when loading weights: {result.missing_keys}")
        if result.unexpected_keys:
            warnings.warn(
                f"Unexpected keys when loading weights: {result.unexpected_keys}")

    # ------------------------------------------------------------------
    # Game interface
    # ------------------------------------------------------------------

    def reset(self):
        """Prepare for a new game (no-op for stateless models)."""

    @torch.inference_mode()
    def act(self, obs):
        """Select an action for the given observation.

        Returns:
            A legal action object from the ``riichienv`` environment
            (e.g. ``Action`` for 4P, ``Action3P`` for 3P), chosen by
            greedy argmax over masked logits.
        """
        feat = self.encoder.encode(obs)
        mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()

        feat_batch = feat.to(self.device).unsqueeze(0)
        mask_t = torch.from_numpy(mask).to(self.device).unsqueeze(0)

        output = self.model(feat_batch)
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits.masked_fill(mask_t == 0, -1e9)
        action_idx = logits.argmax(dim=1).item()

        action = obs.find_action(action_idx)
        if action is not None:
            return action

        # Fallback: first legal action (should be unreachable)
        legals = obs.legal_actions()
        if legals:
            return legals[0]
        raise ValueError(f"No legal action for action_id={action_idx}")
