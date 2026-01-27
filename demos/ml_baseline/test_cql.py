import numpy as np
import torch
from riichienv import RiichiEnv

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

            action_id_map = {}
            mask = torch.from_numpy(np.zeros(82, dtype=np.float32)).to(self.device)
            for legal_action in obs.legal_actions():
                aid = legal_action.encode()
                mask[aid] = 1.0
                action_id_map[aid] = legal_action
            q_values = q_values.masked_fill(mask == 0, -1e9)
            action_selected = q_values.argmax(dim=1).item()
            return action_id_map[action_selected]


def main() -> None:
    env = RiichiEnv(game_mode="4p-red-half")
    obs_dict = env.reset()

    cql_agent = CQLAgent("./cql_model.pth")
    agents = {
        0: cql_agent,
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

    print(env.scores())
    print(env.ranks())


if __name__ == "__main__":
    main()