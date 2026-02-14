"""
Online PPO Trainer with Ray distributed workers.

Supports two modes (controlled by cfg.async_rollout):

Synchronous (async_rollout=False, default):
  dispatch → ray.get → update → sync → dispatch → ...

Double-buffered (async_rollout=True):
  dispatch₀ → ray.get₀ → dispatch₁ → update₀ → sync → ray.get₁ → ...
  Collection N+1 overlaps with PPO update N (1-step stale weights).
  Eval steps fall back to synchronous to ensure fresh weights.
"""
import sys
import os
import threading

import ray
import wandb
import numpy as np
import torch
from loguru import logger

from riichienv_ml.training.ppo_worker import PPOWorker
from riichienv_ml.training.ppo_learner import PPOLearner


def evaluate_parallel(workers, hero_weights, baseline_weights, num_episodes):
    """
    Evaluates hero vs baseline using all workers in parallel.
    Returns (mean_reward, mean_rank, rank_se).
    """
    hero_ref = ray.put(hero_weights)
    baseline_ref = ray.put(baseline_weights)
    ray.get([w.update_weights.remote(hero_ref) for w in workers])
    ray.get([w.update_baseline_weights.remote(baseline_ref) for w in workers])

    eval_futures = [w.evaluate_episodes.remote() for w in workers]
    eval_results = ray.get(eval_futures)

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


def run_ppo_training(cfg):
    """
    Main online PPO training loop.

    Args:
        cfg: OnlineConfig pydantic model.
    """
    python_path = ":".join(sys.path)
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

    learner = PPOLearner(
        device=cfg.device,
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ppo_clip=cfg.ppo_clip,
        ppo_epochs=cfg.ppo_epochs,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        alpha_kl=cfg.alpha_kl,
        alpha_kl_warmup_steps=cfg.alpha_kl_warmup_steps,
        batch_size=cfg.batch_size,
        model_config=model_config,
        model_class=cfg.model_class,
    )

    if cfg.load_model:
        learner.load_weights(cfg.load_model)

    worker_kwargs = dict(
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        num_envs=cfg.num_envs_per_worker,
        model_config=model_config,
        model_class=cfg.model_class,
        encoder_class=cfg.encoder_class,
        grp_model=cfg.grp_model,
        pts_weight=cfg.pts_weight,
    )
    if cfg.worker_device == "cuda":
        workers = [
            PPOWorker.options(num_gpus=cfg.gpu_per_worker).remote(
                i, "cuda", **worker_kwargs,
            )
            for i in range(cfg.num_workers)
        ]
    else:
        workers = [
            PPOWorker.remote(i, "cpu", **worker_kwargs)
            for i in range(cfg.num_workers)
        ]

    # Initial Weight Sync (hero = current policy, baseline = frozen opponents)
    hero_weights = {k: v.cpu() for k, v in learner.get_weights().items()}
    baseline_weights = {k: v.cpu() for k, v in learner.get_weights().items()}
    hero_ref = ray.put(hero_weights)
    baseline_ref = ray.put(baseline_weights)

    for w in workers:
        w.update_weights.remote(hero_ref)
        w.update_baseline_weights.remote(baseline_ref)

    step = 0
    episodes = 0
    wandb.init(project=cfg.wandb_project, config=cfg.model_dump())

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Dry Run Evaluation (parallel)
    logger.info(f"Running dry-run evaluation ({cfg.eval_episodes} episodes, parallel)...")
    try:
        eval_reward, eval_rank, eval_rank_se = evaluate_parallel(
            workers, hero_weights, baseline_weights,
            num_episodes=cfg.eval_episodes)
        logger.info(f"Dry-run evaluation passed: Reward={eval_reward:.2f}, "
                     f"Rank={eval_rank:.2f}\u00b1{eval_rank_se:.2f}")
    except Exception as e:
        logger.error(f"Dry-run evaluation failed: {e}")
        raise e

    total_envs = cfg.num_workers * cfg.num_envs_per_worker

    # For async_rollout: prefetch first collection before entering the loop
    prefetched_futures = None
    if cfg.async_rollout:
        logger.info("Async rollout enabled: using double buffering")
        prefetched_futures = [w.collect_episodes.remote() for w in workers]

    try:
        while step < cfg.num_steps:
            # === 1. Get rollout results ===
            if prefetched_futures is not None:
                # Async mode: results were prefetched
                all_results = ray.get(prefetched_futures)
                prefetched_futures = None
            else:
                # Sync mode: dispatch and wait
                futures = [w.collect_episodes.remote() for w in workers]
                all_results = ray.get(futures)

            # === 2. Aggregate transitions and worker stats ===
            batch_parts = []
            worker_stats_list = []
            for transitions, stats in all_results:
                if transitions:
                    batch_parts.append(transitions)
                if stats:
                    worker_stats_list.append(stats)
            episodes += total_envs

            if not batch_parts:
                # Async mode: dispatch next collection even on empty batch
                if cfg.async_rollout:
                    prefetched_futures = [w.collect_episodes.remote() for w in workers]
                continue

            # Concatenate pre-batched arrays from all workers
            rollout_batch = {
                "features": torch.from_numpy(np.concatenate([b["features"] for b in batch_parts])),
                "masks": torch.from_numpy(np.concatenate([b["mask"] for b in batch_parts])),
                "actions": torch.from_numpy(np.concatenate([b["action"] for b in batch_parts])),
                "old_log_probs": torch.from_numpy(np.concatenate([b["log_prob"] for b in batch_parts])),
                "advantages": torch.from_numpy(np.concatenate([b["advantage"] for b in batch_parts])),
                "returns": torch.from_numpy(np.concatenate([b["return"] for b in batch_parts])),
            }
            n_trans = len(rollout_batch["actions"])

            # === 3. Prefetch next collection (async mode, non-eval steps) ===
            is_eval_step = (step + 1) > 0 and (step + 1) % cfg.eval_interval == 0
            if cfg.async_rollout and not is_eval_step and (step + 1) < cfg.num_steps:
                prefetched_futures = [w.collect_episodes.remote() for w in workers]

            # === 4. PPO update on the full batch (overlaps with collection in async mode) ===
            metrics = learner.update(rollout_batch)
            step += 1

            # Aggregate worker stats
            if worker_stats_list:
                for key in worker_stats_list[0]:
                    vals = [s[key] for s in worker_stats_list if key in s]
                    metrics[f"rollout/{key}"] = float(np.mean(vals))

            # Logging
            log_msg = (
                f"Step {step}: loss={metrics['loss']:.4f}, "
                f"pi={metrics['policy_loss']:.4f}, v={metrics['value_loss']:.4f}, "
                f"ent={metrics['entropy']:.4f}, kl={metrics['approx_kl']:.4f}, "
                f"kl_ref={metrics.get('kl_ref', 0):.4f}, "
                f"clip={metrics['clip_frac']:.3f}, "
                f"adv={metrics.get('adv/raw_mean', 0):.3f}\u00b1{metrics.get('adv/raw_std', 0):.3f}, "
                f"ret={metrics.get('return/mean', 0):.3f}, "
                f"ev={metrics.get('explained_variance', 0):.3f}, "
                f"ep={episodes}, trans={n_trans}"
            )
            if worker_stats_list:
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
            logger.info(log_msg)
            wandb.log(metrics, step=step)

            # === 5. Periodic Evaluation & Snapshot ===
            if step > 0 and step % cfg.eval_interval == 0:
                logger.info(f"Step {step}: Saving snapshot and Evaluating...")
                sys.stdout.flush()

                save_path = f"{cfg.checkpoint_dir}/model_{step}.pth"
                save_weights = {k: v.cpu().clone() for k, v in learner.get_weights().items()}
                threading.Thread(target=torch.save, args=(save_weights, save_path), daemon=True).start()
                logger.info(f"Saved snapshot to {save_path} (async)")

                try:
                    hw = {k: v.cpu() for k, v in learner.get_weights().items()}
                    eval_reward, eval_rank, eval_rank_se = evaluate_parallel(
                        workers, hw, baseline_weights,
                        num_episodes=cfg.eval_episodes)
                    logger.info(f"Evaluation (vs Baseline): Reward={eval_reward:.2f}, "
                                f"Rank={eval_rank:.2f}\u00b1{eval_rank_se:.2f}")
                    wandb.log({
                        "eval/reward": eval_reward,
                        "eval/rank": eval_rank,
                        "eval/rank_se": eval_rank_se,
                    }, step=step)
                except Exception as e:
                    logger.error(f"Evaluation failed at step {step}: {e}")

            # === 6. Sync updated weights to ALL workers ===
            weights = {k: v.cpu() for k, v in learner.get_weights().items()}

            has_nan = False
            for name, param in weights.items():
                if torch.isnan(param).any():
                    logger.error(f"NaN detected in learner weights {name} at step {step}")
                    has_nan = True
            if has_nan:
                logger.critical("Cannot sync NaN weights to workers. Stopping training.")
                break

            weight_ref = ray.put(weights)
            for w in workers:
                w.update_weights.remote(weight_ref)

            # === 7. After eval + weight sync, dispatch next collection (async mode) ===
            if cfg.async_rollout and prefetched_futures is None and step < cfg.num_steps:
                prefetched_futures = [w.collect_episodes.remote() for w in workers]

    except KeyboardInterrupt:
        logger.info("Stopping...")

    # Final Snapshot and Evaluation
    logger.info(f"Final Step {step} (episodes={episodes}): Saving snapshot and Evaluating...")
    sys.stdout.flush()

    save_path = f"{cfg.checkpoint_dir}/model_{step}.pth"
    torch.save(learner.get_weights(), save_path)
    logger.info(f"Saved snapshot to {save_path}")

    try:
        hw = {k: v.cpu() for k, v in learner.get_weights().items()}
        eval_reward, eval_rank, eval_rank_se = evaluate_parallel(
            workers, hw, baseline_weights,
            num_episodes=cfg.eval_episodes)
        logger.info(f"Final Evaluation (vs Baseline): Reward={eval_reward:.2f}, "
                    f"Rank={eval_rank:.2f}\u00b1{eval_rank_se:.2f}")
        wandb.log({
            "eval/reward": eval_reward,
            "eval/rank": eval_rank,
            "eval/rank_se": eval_rank_se,
        }, step=step)
    except Exception as e:
        logger.error(f"Final Evaluation failed: {e}")

    ray.shutdown()
