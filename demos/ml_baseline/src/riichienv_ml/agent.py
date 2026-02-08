import torch
import numpy as np

from riichienv import Action
from riichienv_ml.config import load_config, import_class


class RiichiEnvAgent:
    """Agent that loads a model from a riichienv_ml config YAML and checkpoint."""

    def __init__(self, config_path: str, model_path: str, device_str="cuda"):
        self.device = torch.device(device_str)

        cfg = load_config(config_path).online
        model_config = cfg.model.model_dump()

        # Build model directly from config model_class
        ModelClass = import_class(cfg.model_class)
        self.model = ModelClass(**model_config).to(self.device)

        # Load weights with format auto-detection
        state = torch.load(model_path, map_location=self.device)
        has_actor = any(k.startswith("actor_head.") for k in state.keys())
        has_head = any(k.startswith("head.") for k in state.keys())

        if has_head and not has_actor:
            # QNetwork checkpoint -> ActorCriticNetwork: map head -> actor_head
            new_state = {}
            for k, v in state.items():
                if k.startswith("head."):
                    new_state[k.replace("head.", "actor_head.")] = v
                else:
                    new_state[k] = v
            self.model.load_state_dict(new_state, strict=False)
            print(f"Loaded QNetwork checkpoint (head -> actor_head)")
        else:
            self.model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint from {model_path}")

        self.model.eval()

        # Resolve encoder from config
        self.encoder = import_class(cfg.encoder_class)

    def act(self, obs):
        with torch.no_grad():
            feat = self.encoder.encode(obs)
            mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()

            feat_batch = feat.to(self.device).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).to(self.device).unsqueeze(0)

            output = self.model(feat_batch)
            # Handle both QNetwork (tensor) and ActorCriticNetwork (tuple)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            logits = logits.masked_fill(mask_tensor == 0, -1e9)
            action_idx = logits.argmax(dim=1).item()

            found_action: Action | None = obs.find_action(action_idx)
            if found_action is None:
                raise ValueError(
                    f"No legal action found for selected action id {action_idx}"
                )
            return found_action