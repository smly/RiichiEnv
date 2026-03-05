from riichienv import RiichiEnv
from riichienv_ml.agents import Agent

CONFIG_PATH = "src/riichienv_ml/configs/3p/bc_logs.yml"
CONFIG3_PATH = "src/riichienv_ml/configs/3p/bc_ppo.yml"
MODEL_PATH = "/data/workspace/riichienv-ml/3p/bc_logs_2M.pth"
MODEL2_PATH = "/data/workspace/riichienv-ml/3p/bc_logs_2M.pth"
MODEL3_PATH = "/data/workspace/riichienv-ml/3p/bc_ppo/checkpoints/model_500.pth"

agent = Agent(CONFIG_PATH, MODEL_PATH, device="cuda")
agent2 = Agent(CONFIG_PATH, MODEL2_PATH, device="cuda")
agent3 = Agent(CONFIG3_PATH, MODEL3_PATH, device="cuda")
agents = {0: agent, 1: agent2, 2: agent3}

env = RiichiEnv(game_mode="3p-red-half")
obs_dict = env.reset()

while not env.done():
    actions = {pid: agents[pid].act(obs) for pid, obs in obs_dict.items()}
    obs_dict = env.step(actions)

print(env.ranks(), env.scores())
