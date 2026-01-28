import numpy as np
import torch
from riichienv import RiichiEnv
from riichienv.agents import RandomAgent

from cql_model import QNetwork


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


def main() -> None:
    env = RiichiEnv(game_mode="4p-red-half")
    obs_dict = env.reset()

    cql_agent = CQLAgent("./cql_model.pth")
    random_agent = RandomAgent()
    agents = {
        0: cql_agent,
        1: random_agent,
        2: random_agent,
        3: random_agent,
    }

    while not env.done():
        steps = {
            player_id: agents[player_id].act(obs)
            for player_id, obs in obs_dict.items()
        }
        obs_dict = env.step(steps)

    print(env.scores())
    print(env.ranks())


if __name__ == "__main__":
    main()