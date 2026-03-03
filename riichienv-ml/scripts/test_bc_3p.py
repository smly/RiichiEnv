from riichienv import RiichiEnv
from riichienv_ml.agents import Agent

CONFIG_PATH = "src/riichienv_ml/configs/3p/bc_logs.yml"
MODEL_PATH = "/data/workspace/riichienv-ml/3p/bc_logs.pth"

agent = Agent(CONFIG_PATH, MODEL_PATH, device="cuda")
agents = {0: agent, 1: agent, 2: agent}

env = RiichiEnv(game_mode="3p-red-half")
obs_dict = env.reset()
print(obs_dict)
while not env.done():
    actions = {pid: agents[pid].act(obs) for pid, obs in obs_dict.items()}
    print(actions)
    obs_dict = env.step(actions)
    print(obs_dict)

print(env.ranks(), env.scores())