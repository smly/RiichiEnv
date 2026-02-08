"""
Online DQN + CQL Trainer with Ray distributed workers.
"""
import sys
import os

import ray
import wandb
import numpy as np
import torch
from riichienv import RiichiEnv

from riichienv_ml.config import import_class
from riichienv_ml.training.ray_actor import MahjongWorker
from riichienv_ml.training.learner import MahjongLearner
from riichienv_ml.training.buffer import GlobalReplayBuffer


def evaluate_vs_baseline(hero_model, baseline_model, device, encoder, num_episodes=30):
    """
    Evaluates Hero Model (Player 0) vs Baseline Model (Players 1-3).
    Both use greedy argmax Q-value action selection.
    """
    hero_rewards = []
    hero_ranks = []

    hero_model.eval()
    baseline_model.eval()

    for _ in range(num_episodes):
        env = RiichiEnv(game_mode="4p-red-half")
        obs_dict = env.reset()

        while not env.done():
            steps = {}
            for pid, obs in obs_dict.items():
                model = hero_model if pid == 0 else baseline_model

                feat = encoder.encode(obs)
                mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()

                feat_t = feat.to(device).unsqueeze(0)
                mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)

                with torch.no_grad():
                    q_values = model(feat_t)
                    q_values = q_values.masked_fill(mask_t == 0, -1e9)
                    action_idx = q_values.argmax(dim=1).item()

                found_action = obs.find_action(action_idx)
                if found_action is None:
                    found_action = obs.legal_actions()[0]

                steps[pid] = found_action

            obs_dict = env.step(steps)

        ranks = env.ranks()
        rank = ranks[0]
        reward = 0.0
        if rank == 1: reward = 10.0
        elif rank == 2: reward = 4.0
        elif rank == 3: reward = -4.0
        elif rank == 4: reward = -10.0

        hero_rewards.append(reward)
        hero_ranks.append(rank)

    return np.mean(hero_rewards), np.mean(hero_ranks)


def run_training(cfg):
    """
    Main online DQN training loop.

    Args:
        cfg: OnlineConfig pydantic model.
    """
    python_path = ":".join(sys.path)
    # Point to src/ so Ray ships the entire riichienv_ml package to workers
    src_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

    runtime_env = {
        "working_dir": src_dir,
        "excludes": [".git", ".venv", "wandb", "__pycache__", "pyproject.toml", "uv.lock"],
        "env_vars": {
            "PYTHONPATH": python_path,
            "PATH": os.environ["PATH"]
        }
    }

    ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

    model_config = cfg.model.model_dump()
    encoder = import_class(cfg.encoder_class)

    learner = MahjongLearner(
        device=cfg.device,
        lr=cfg.lr,
        alpha_cql_init=cfg.alpha_cql_init,
        alpha_cql_final=cfg.alpha_cql_final,
        gamma=cfg.gamma,
        model_config=model_config,
        model_class=cfg.model_class,
    )
    baseline_learner = None

    if cfg.load_model:
        learner.load_cql_weights(cfg.load_model)
        baseline_learner = MahjongLearner(
            device=cfg.device, model_config=model_config, model_class=cfg.model_class,
        )
        baseline_learner.load_cql_weights(cfg.load_model)
        baseline_learner.model.eval()

    buffer = GlobalReplayBuffer(
        capacity=cfg.capacity,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )

    worker_kwargs = dict(
        gamma=cfg.gamma,
        exploration=cfg.exploration,
        epsilon=cfg.epsilon_start,
        boltzmann_epsilon=cfg.boltzmann_epsilon,
        boltzmann_temp=cfg.boltzmann_temp_start,
        top_p=cfg.top_p,
        model_config=model_config, model_class=cfg.model_class,
        encoder_class=cfg.encoder_class,
    )
    if cfg.worker_device == "cuda":
        workers = [
            MahjongWorker.options(num_gpus=cfg.gpu_per_worker).remote(
                i, "cuda", **worker_kwargs,
            )
            for i in range(cfg.num_workers)
        ]
    else:
        workers = [
            MahjongWorker.remote(i, "cpu", **worker_kwargs)
            for i in range(cfg.num_workers)
        ]

    # Initial Weight Sync
    weights = {k: v.cpu() for k, v in learner.get_weights().items()}
    weight_ref = ray.put(weights)

    for w in workers:
        w.update_weights.remote(weight_ref)

    future_to_worker = {w.collect_episode.remote(): i for i, w in enumerate(workers)}

    step = 0  # counts gradient updates
    episodes = 0
    last_log_step = 0
    last_eval_step = 0
    last_sync_step = 0
    wandb.init(project=cfg.wandb_project, config=cfg.model_dump())

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Dry Run Evaluation
    if baseline_learner is not None:
        print("Running dry-run evaluation (30 episodes)...")
        try:
            evaluate_vs_baseline(learner.model, baseline_learner.model, cfg.device, encoder, num_episodes=30)
            print("Dry-run evaluation passed.")
        except Exception as e:
            print(f"Dry-run evaluation failed: {e}")
            raise e

    try:
        while step < cfg.num_steps:
            ready_ids, _ = ray.wait(list(future_to_worker.keys()), num_returns=1)
            future = ready_ids[0]
            worker_idx = future_to_worker.pop(future)

            transitions = ray.get(future)
            buffer.add(transitions)
            episodes += 1

            # Train - Multiple gradient steps per episode for better data efficiency.
            if len(buffer) > cfg.batch_size:
                num_updates = max(1, len(transitions) // cfg.batch_size)
                for _ in range(num_updates):
                    if step >= cfg.num_steps:
                        break

                    # Periodic Evaluation & Snapshot
                    if step > 0 and step - last_eval_step >= cfg.eval_interval:
                        last_eval_step = step
                        print(f"Step {step}: Saving snapshot and Evaluating...")
                        sys.stdout.flush()

                        save_path = f"{cfg.checkpoint_dir}/model_{step}.pth"
                        torch.save(learner.get_weights(), save_path)
                        print(f"Saved snapshot to {save_path}")

                        if baseline_learner is not None:
                            try:
                                eval_reward, eval_rank = evaluate_vs_baseline(
                                    learner.model, baseline_learner.model, cfg.device, encoder, num_episodes=30
                                )
                                print(f"Evaluation (vs Baseline): Reward={eval_reward:.2f}, Rank={eval_rank:.2f}")
                                wandb.log({
                                    "eval/reward": eval_reward,
                                    "eval/rank": eval_rank
                                }, step=step)
                            except Exception as e:
                                print(f"Evaluation failed at step {step}: {e}")

                    batch = buffer.sample(cfg.batch_size)
                    metrics = learner.update(batch, max_steps=cfg.num_steps)
                    step += 1

                    # Logging
                    if step - last_log_step >= 200:
                        last_log_step = step
                        print(f"Step {step}: {metrics}, ep={episodes}, buf={len(buffer)}, trans={len(transitions)}")
                        wandb.log(metrics, step=step)

            # Exploration scheduling
            progress = min(1.0, step / cfg.num_steps)

            # Re-dispatch worker with weight sync
            if step - last_sync_step >= cfg.weight_sync_freq:
                last_sync_step = step
                weights = {k: v.cpu() for k, v in learner.get_weights().items()}

                has_nan = False
                for name, param in weights.items():
                    if torch.isnan(param).any():
                        print(f"ERROR: NaN detected in learner weights {name} at step {step}")
                        has_nan = True

                if has_nan:
                    print(f"FATAL: Cannot sync NaN weights to workers. Stopping training.")
                    break

                weight_ref = ray.put(weights)
                if cfg.exploration == "boltzmann":
                    temp = cfg.boltzmann_temp_start + progress * (cfg.boltzmann_temp_final - cfg.boltzmann_temp_start)
                else:
                    epsilon = cfg.epsilon_start + progress * (cfg.epsilon_final - cfg.epsilon_start)

                for w in workers:
                    w.update_weights.remote(weight_ref)
                    if cfg.exploration == "boltzmann":
                        w.set_boltzmann_temp.remote(temp)
                    else:
                        w.set_epsilon.remote(epsilon)

            new_future = workers[worker_idx].collect_episode.remote()
            future_to_worker[new_future] = worker_idx

    except KeyboardInterrupt:
        print("Stopping...")

    # Final Snapshot and Evaluation
    print(f"Final Step {step} (episodes={episodes}): Saving snapshot and Evaluating...")
    sys.stdout.flush()

    save_path = f"{cfg.checkpoint_dir}/model_{step}.pth"
    torch.save(learner.get_weights(), save_path)
    print(f"Saved snapshot to {save_path}")

    if baseline_learner is not None:
        try:
            eval_reward, eval_rank = evaluate_vs_baseline(
                learner.model, baseline_learner.model, cfg.device, encoder, num_episodes=30
            )
            print(f"Final Evaluation (vs Baseline): Reward={eval_reward:.2f}, Rank={eval_rank:.2f}")
            wandb.log({
                "eval/reward": eval_reward,
                "eval/rank": eval_rank
            }, step=step)
        except Exception as e:
            print(f"Final Evaluation failed: {e}")

    ray.shutdown()
