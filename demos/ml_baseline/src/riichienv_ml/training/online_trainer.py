"""
Online DQN Trainer — Synchronous On-Policy (Mortal-style).

Each round:
  1. All workers collect episodes in parallel
  2. Wait for all workers to finish
  3. Concatenate transitions, shuffle, train 1 pass (each datum used once)
  4. Sync weights to all workers
  5. Repeat
"""
import sys
import os
import threading

import ray
import wandb
import numpy as np
import torch

from riichienv_ml.training.ray_actor import RVWorker
from riichienv_ml.training.learner import MahjongLearner
from riichienv_ml.training.buffer import OnPolicyBuffer


def evaluate_parallel(workers, hero_weights, baseline_weights, num_episodes):
    """
    Evaluates hero vs baseline using all workers in parallel.
    Returns (mean_reward, mean_rank, rank_se).
    """
    # Sync hero + baseline weights to all workers
    hero_ref = ray.put(hero_weights)
    baseline_ref = ray.put(baseline_weights)
    ray.get([w.update_weights.remote(hero_ref) for w in workers])
    ray.get([w.update_baseline_weights.remote(baseline_ref) for w in workers])

    # Dispatch evaluation across all workers (each runs num_envs games)
    eval_futures = [w.evaluate_episodes.remote() for w in workers]
    eval_results = ray.get(eval_futures)

    # Aggregate results (each worker returns list of (reward, rank))
    all_rewards = []
    all_ranks = []
    for worker_results in eval_results:
        for reward, rank in worker_results:
            all_rewards.append(reward)
            all_ranks.append(rank)
            if len(all_rewards) >= num_episodes:
                break
        if len(all_rewards) >= num_episodes:
            break

    all_rewards = all_rewards[:num_episodes]
    all_ranks = all_ranks[:num_episodes]

    mean_reward = float(np.mean(all_rewards))
    mean_rank = float(np.mean(all_ranks))
    rank_se = float(np.std(all_ranks) / np.sqrt(len(all_ranks)))

    return mean_reward, mean_rank, rank_se


def run_training(cfg):
    """
    Main online DQN training loop (synchronous on-policy, Mortal-style).

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

    learner = MahjongLearner(
        device=cfg.device,
        lr=cfg.lr,
        lr_min=cfg.lr_min,
        num_steps=cfg.num_steps,
        max_grad_norm=cfg.max_grad_norm,
        alpha_cql_init=cfg.alpha_cql_init,
        alpha_cql_final=cfg.alpha_cql_final,
        alpha_kl=cfg.alpha_kl,
        gamma=cfg.gamma,
        weight_decay=cfg.weight_decay,
        aux_weight=cfg.aux_weight,
        entropy_coef=cfg.entropy_coef,
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

    buffer = OnPolicyBuffer(device=cfg.device)

    worker_kwargs = dict(
        gamma=cfg.gamma,
        exploration=cfg.exploration,
        epsilon=cfg.epsilon_start,
        boltzmann_epsilon=cfg.boltzmann_epsilon,
        boltzmann_temp=cfg.boltzmann_temp_start,
        top_p=cfg.top_p,
        num_envs=cfg.num_envs_per_worker,
        model_config=model_config, model_class=cfg.model_class,
        encoder_class=cfg.encoder_class,
        grp_model=cfg.grp_model,
        pts_weight=cfg.pts_weight,
        collect_hero_only=cfg.collect_hero_only,
    )
    if cfg.worker_device == "cuda":
        workers = [
            RVWorker.options(num_gpus=cfg.gpu_per_worker).remote(
                i, "cuda", **worker_kwargs,
            )
            for i in range(cfg.num_workers)
        ]
    else:
        workers = [
            RVWorker.remote(i, "cpu", **worker_kwargs)
            for i in range(cfg.num_workers)
        ]

    # Initial Weight Sync (hero + baseline)
    hero_weights = {k: v.cpu() for k, v in learner.get_weights().items()}
    hero_ref = ray.put(hero_weights)

    baseline_weights = None
    if baseline_learner is not None:
        baseline_weights = {k: v.cpu() for k, v in baseline_learner.get_weights().items()}
        baseline_ref = ray.put(baseline_weights)

    for w in workers:
        w.update_weights.remote(hero_ref)
        if baseline_weights is not None:
            w.update_baseline_weights.remote(baseline_ref)

    step = 0  # counts gradient updates
    episodes = 0
    round_num = 0
    last_log_step = 0
    last_eval_step = 0
    wandb.init(project=cfg.wandb_project, config=cfg.model_dump())

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Dry Run Evaluation
    if baseline_learner is not None:
        print(f"Running dry-run evaluation ({cfg.eval_episodes} episodes, parallel)...")
        try:
            eval_reward, eval_rank, eval_rank_se = evaluate_parallel(
                workers, hero_weights, baseline_weights,
                num_episodes=cfg.eval_episodes)
            print(f"Dry-run evaluation passed: Reward={eval_reward:.2f}, "
                  f"Rank={eval_rank:.2f}\u00b1{eval_rank_se:.2f}")
        except Exception as e:
            print(f"Dry-run evaluation failed: {e}")
            raise e

    worker_stats_agg = []

    try:
        while step < cfg.num_steps:
            # === 1. Dispatch collect_episodes to ALL workers ===
            futures = [w.collect_episodes.remote() for w in workers]

            # === 2. Wait for ALL workers to complete ===
            results = ray.get(futures)

            # === 3. Aggregate worker stats and count episodes ===
            worker_transitions = []
            for transitions, worker_stats in results:
                if transitions:
                    worker_transitions.append(transitions)
                episodes += cfg.num_envs_per_worker
                if worker_stats:
                    worker_stats_agg.append(worker_stats)

            if not worker_transitions:
                round_num += 1
                continue

            # === 4. Load into on-policy buffer (concatenate + shuffle) ===
            buffer.set_data(worker_transitions)
            round_transitions = buffer.size

            # === 5. Train: single pass over all data (each datum used once) ===
            for batch in buffer.iter_batches(cfg.batch_size):
                if step >= cfg.num_steps:
                    break

                metrics = learner.update(batch, max_steps=cfg.num_steps)
                step += 1

                # Logging
                if step - last_log_step >= 200:
                    last_log_step = step
                    log_msg = (
                        f"Step {step} (round {round_num}): "
                        f"loss={metrics['loss']:.4f}, "
                        f"kl={metrics.get('kl', 0):.4f}, "
                        f"cql={metrics['cql']:.4f}, td={metrics['td']:.4f}, "
                        f"q={metrics['q_mean']:.3f}\u00b1{metrics.get('q_std', 0):.3f}, "
                        f"ent={metrics.get('q_entropy', 0):.3f}, "
                        f"adv_std={metrics.get('advantage_std', 0):.4f}, "
                        f"lr={metrics.get('lr', 0):.1e}, "
                        f"gn={metrics.get('grad_norm', 0):.2f}, "
                        f"ep={episodes}, trans={round_transitions}"
                    )
                    if worker_stats_agg:
                        for key in worker_stats_agg[0]:
                            vals = [s[key] for s in worker_stats_agg if key in s]
                            metrics[f"rollout/{key}"] = float(np.mean(vals))
                        log_msg += (
                            f", rew={metrics.get('rollout/reward_mean', 0):.2f}"
                            f", rank={metrics.get('rollout/rank_mean', 0):.2f}"
                        )
                        if "rollout/kyoku_reward_mean" in metrics:
                            log_msg += (
                                f", k_rew={metrics['rollout/kyoku_reward_mean']:.3f}"
                                f"\u00b1{metrics.get('rollout/kyoku_reward_std', 0):.3f}"
                                f", k_len={metrics.get('rollout/kyoku_length_mean', 0):.1f}"
                            )
                        worker_stats_agg = []
                    metrics["buffer/round_transitions"] = round_transitions
                    metrics["round"] = round_num
                    print(log_msg)
                    wandb.log(metrics, step=step)

            # === 6. Clear buffer (data used exactly once) ===
            buffer.clear()

            # === 7. Periodic Evaluation & Snapshot ===
            if step > 0 and step - last_eval_step >= cfg.eval_interval:
                last_eval_step = step
                print(f"Step {step}: Saving snapshot and Evaluating...")
                sys.stdout.flush()

                save_path = f"{cfg.checkpoint_dir}/model_{step}.pth"
                save_weights = {k: v.cpu().clone() for k, v in learner.get_weights().items()}
                threading.Thread(target=torch.save, args=(save_weights, save_path), daemon=True).start()
                print(f"Saving snapshot to {save_path} (async)")

                if baseline_learner is not None:
                    try:
                        hw = {k: v.cpu() for k, v in learner.get_weights().items()}
                        eval_reward, eval_rank, eval_rank_se = evaluate_parallel(
                            workers, hw, baseline_weights,
                            num_episodes=cfg.eval_episodes)
                        print(f"Evaluation (vs Baseline): Reward={eval_reward:.2f}, "
                              f"Rank={eval_rank:.2f}\u00b1{eval_rank_se:.2f}")
                        wandb.log({
                            "eval/reward": eval_reward,
                            "eval/rank": eval_rank,
                            "eval/rank_se": eval_rank_se,
                        }, step=step)
                    except Exception as e:
                        print(f"Evaluation failed at step {step}: {e}")

            # === 8. Sync weights to all workers ===
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

            # Exploration scheduling
            progress = min(1.0, step / cfg.num_steps)
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

            round_num += 1

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
            hw = {k: v.cpu() for k, v in learner.get_weights().items()}
            eval_reward, eval_rank, eval_rank_se = evaluate_parallel(
                workers, hw, baseline_weights,
                num_episodes=cfg.eval_episodes)
            print(f"Final Evaluation (vs Baseline): Reward={eval_reward:.2f}, "
                  f"Rank={eval_rank:.2f}\u00b1{eval_rank_se:.2f}")
            wandb.log({
                "eval/reward": eval_reward,
                "eval/rank": eval_rank,
                "eval/rank_se": eval_rank_se,
            }, step=step)
        except Exception as e:
            print(f"Final Evaluation failed: {e}")

    ray.shutdown()
