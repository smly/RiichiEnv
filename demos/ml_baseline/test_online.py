import numpy as np
import torch
from riichienv import RiichiEnv, Action
from torch.distributions import Categorical

from cql_model import QNetwork
from unified_model import UnifiedNetwork


class CQLAgent:
    def __init__(self, model_path, device_str="cuda"):
        self.device = torch.device(device_str)
        self.model = QNetwork(in_channels=46, num_actions=82).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def act(self, obs):
        with torch.no_grad():
            feat = np.frombuffer(obs.encode(), dtype=np.float32).reshape(46, 34).copy()
            q_values = self.model(torch.from_numpy(feat).unsqueeze(0).to(self.device))

            mask = torch.from_numpy(np.frombuffer(obs.mask(), dtype=np.uint8).copy()).to(self.device)
            q_values = q_values.masked_fill(mask == 0, -1e9)
            action_selected = q_values.argmax(dim=1).item()

            found_action: Action | None = obs.find_action(action_selected)
            if found_action is None:
                raise ValueError(
                    f"No legal action found for selected action id {action_selected}"
                )
            return found_action


class PPOAgent:
    def __init__(self, model_path, device_str="cuda"):
        self.device = torch.device(device_str)
        self.model = UnifiedNetwork(in_channels=46, num_actions=82).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def act(self, obs):
        with torch.no_grad():
            feat = np.frombuffer(obs.encode(), dtype=np.float32).reshape(46, 34).copy()
            logits, q_values = self.model(torch.from_numpy(feat).unsqueeze(0).to(self.device))

            mask = torch.from_numpy(np.frombuffer(obs.mask(), dtype=np.uint8).copy()).to(self.device)
            logits = logits.masked_fill(mask == 0, -1e9)
            dist = Categorical(logits=logits)
            action = dist.sample()
            action_idx = action.item()

            found_action: Action | None = obs.find_action(action_idx)
            if found_action is None:
                raise ValueError(
                    f"No legal action found for selected action id {action_idx}"
                )
            return found_action


def main() -> None:
    ranks = []
    env = RiichiEnv(game_mode="4p-red-half")

    for i in range(10):
        obs_dict = env.reset(
            scores=[25000, 25000, 25000, 25000],
            round_wind=0,
            oya=0,
            honba=0,
            kyotaku=0
        )

        cql_agent = CQLAgent("./cql_model.pth")
        ppo_agent = PPOAgent("./checkpoints/model_70000.pth")
        agents = {
            0: ppo_agent,
            1: cql_agent,
            2: cql_agent,
            3: cql_agent,
        }

        while not env.done():
            steps = {
                player_id: agents[player_id].act(obs)
                for player_id, obs in obs_dict.items()
            }
            obs_dict = env.step(steps)
            if not obs_dict:
                print("ERROR")

        print(env.scores())
        print(env.ranks())
        ranks.append(env.ranks()[0])
        print(env.mjai_log[:2])

    print("Average ranks:", np.mean(ranks))


if __name__ == "__main__":
    main()