"""
Online PPO Trainer with Ray distributed workers.
"""
import sys
import os

import ray
import wandb
import numpy as np
import torch
from loguru import logger
from riichienv import RiichiEnv

from riichienv_ml.config import import_class
from riichienv_ml.training.ppo_worker import PPOWorker
from riichienv_ml.training.ppo_learner import PPOLearner


def evaluate_vs_baseline(hero_model, baseline_model, device, encoder, num_episodes=100):
    """
    Evaluates Hero Model (Player 0) vs Baseline Model (Players 1-3).
    Hero uses ActorCriticNetwork (greedy argmax on logits).
    Baseline uses QNetwork (greedy argmax on Q-values).
    Returns (mean_reward, mean_rank, rank_standard_error).
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
                feat = encoder.encode(obs)
                mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()

                feat_t = feat.to(device).unsqueeze(0)
                mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)

                with torch.no_grad():
                    if pid == 0:
                        # Hero: ActorCriticNetwork
                        output = hero_model(feat_t)
                        if isinstance(output, tuple):
                            logits = output[0]
                        else:
                            logits = output
                        logits = logits.masked_fill(mask_t == 0, -1e9)
                        action_idx = logits.argmax(dim=1).item()
                    else:
                        # Baseline: QNetwork or ActorCriticNetwork
                        output = baseline_model(feat_t)
                        if isinstance(output, tuple):
                            q_values = output[0]
                        else:
                            q_values = output
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

    rank_se = np.std(hero_ranks) / np.sqrt(num_episodes)
    return np.mean(hero_rewards), np.mean(hero_ranks), rank_se


def run_ppo_training(cfg):
    """
    Main online PPO training loop.

    Key difference from DQN: on-policy training. Workers collect trajectories,
    learner trains on them immediately (no replay buffer), then workers collect
    new trajectories with updated policy.
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
    encoder = import_class(cfg.encoder_class)

    learner = PPOLearner(
        device=cfg.device,
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ppo_clip=cfg.ppo_clip,
        ppo_epochs=cfg.ppo_epochs,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        model_config=model_config,
        model_class=cfg.model_class,
    )

    baseline_model = None
    if cfg.load_model:
        learner.load_weights(cfg.load_model)
        # Load baseline for evaluation
        baseline_class = import_class(cfg.model_class)
        baseline_model = baseline_class(**model_config).to(cfg.device)
        state = torch.load(cfg.load_model, map_location=cfg.device)
        # Handle QNetwork -> ActorCriticNetwork mapping for baseline
        has_head = any(k.startswith("head.") for k in state.keys())
        if has_head and not any(k.startswith("actor_head.") for k in state.keys()):
            new_state = {}
            for k, v in state.items():
                if k.startswith("head."):
                    new_state[k.replace("head.", "actor_head.")] = v
                else:
                    new_state[k] = v
            baseline_model.load_state_dict(new_state, strict=False)
        else:
            baseline_model.load_state_dict(state, strict=False)
        baseline_model.eval()
        logger.info(f"Loaded baseline model from {cfg.load_model}")

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
    weights = {k: v.cpu() for k, v in learner.get_weights().items()}
    weight_ref = ray.put(weights)
    for w in workers:
        w.update_weights.remote(weight_ref)
        w.update_baseline_weights.remote(weight_ref)

    step = 0
    episodes = 0
    wandb.init(project=cfg.wandb_project, config=cfg.model_dump())

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Dry Run Evaluation
    if baseline_model is not None:
        logger.info(f"Running dry-run evaluation ({cfg.eval_episodes} episodes)...")
        try:
            evaluate_vs_baseline(learner.model, baseline_model, cfg.device, encoder,
                                 num_episodes=cfg.eval_episodes)
            logger.info("Dry-run evaluation passed.")
        except Exception as e:
            logger.error(f"Dry-run evaluation failed: {e}")
            raise e

    num_workers = cfg.num_workers
    total_envs = num_workers * cfg.num_envs_per_worker

    try:
        while step < cfg.num_steps:
            # === Phase 1: Dispatch all workers with current policy ===
            futures = [w.collect_episodes.remote() for w in workers]

            # === Phase 2: Wait for ALL workers to finish ===
            all_results = ray.get(futures)

            # === Phase 3: Aggregate all transitions and worker stats ===
            all_transitions = []
            worker_stats_list = []
            for transitions, stats in all_results:
                all_transitions.extend(transitions)
                if stats:
                    worker_stats_list.append(stats)
            episodes += total_envs

            if not all_transitions:
                continue

            n_trans = len(all_transitions)
            rollout_batch = {
                "features": torch.from_numpy(np.stack([t["features"] for t in all_transitions])),
                "masks": torch.from_numpy(np.stack([t["mask"] for t in all_transitions])),
                "actions": torch.from_numpy(np.array([t["action"] for t in all_transitions])),
                "old_log_probs": torch.from_numpy(np.array([t["log_prob"] for t in all_transitions], dtype=np.float32)),
                "advantages": torch.from_numpy(np.array([t["advantage"] for t in all_transitions], dtype=np.float32)),
                "returns": torch.from_numpy(np.array([t["return"] for t in all_transitions], dtype=np.float32)),
            }

            # === Phase 4: PPO update on the full batch ===
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
                f"clip={metrics['clip_frac']:.3f}, "
                f"adv={metrics.get('adv/raw_mean', 0):.3f}\u00b1{metrics.get('adv/raw_std', 0):.3f}, "
                f"ret={metrics.get('return/mean', 0):.3f}, "
                f"ev={metrics.get('explained_variance', 0):.3f}, "
                f"ep={episodes}, trans={n_trans}"
            )
            if worker_stats_list:
                log_msg += (
                    f", rew={metrics.get('rollout/reward_mean', 0):.2f}, "
                    f"rank={metrics.get('rollout/rank_mean', 0):.2f}"
                )
                if "rollout/kyoku_reward_mean" in metrics:
                    log_msg += (
                        f", k_rew={metrics['rollout/kyoku_reward_mean']:.3f}"
                        f"\u00b1{metrics.get('rollout/kyoku_reward_std', 0):.3f}"
                        f", k_len={metrics.get('rollout/kyoku_length_mean', 0):.1f}"
                    )
            logger.info(log_msg)
            wandb.log(metrics, step=step)

            # Periodic Evaluation & Snapshot
            if step > 0 and step % cfg.eval_interval == 0:
                logger.info(f"Step {step}: Saving snapshot and Evaluating...")
                sys.stdout.flush()

                save_path = f"{cfg.checkpoint_dir}/model_{step}.pth"
                torch.save(learner.get_weights(), save_path)
                logger.info(f"Saved snapshot to {save_path}")

                if baseline_model is not None:
                    try:
                        eval_reward, eval_rank, eval_rank_se = evaluate_vs_baseline(
                            learner.model, baseline_model, cfg.device, encoder,
                            num_episodes=cfg.eval_episodes
                        )
                        logger.info(f"Evaluation (vs Baseline): Reward={eval_reward:.2f}, "
                                    f"Rank={eval_rank:.2f}\u00b1{eval_rank_se:.2f}")
                        wandb.log({
                            "eval/reward": eval_reward,
                            "eval/rank": eval_rank,
                            "eval/rank_se": eval_rank_se,
                        }, step=step)
                    except Exception as e:
                        logger.error(f"Evaluation failed at step {step}: {e}")

            # === Phase 5: Sync updated weights to ALL workers ===
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

    except KeyboardInterrupt:
        logger.info("Stopping...")

    # Final Snapshot and Evaluation
    logger.info(f"Final Step {step} (episodes={episodes}): Saving snapshot and Evaluating...")
    sys.stdout.flush()

    save_path = f"{cfg.checkpoint_dir}/model_{step}.pth"
    torch.save(learner.get_weights(), save_path)
    logger.info(f"Saved snapshot to {save_path}")

    if baseline_model is not None:
        try:
            eval_reward, eval_rank, eval_rank_se = evaluate_vs_baseline(
                learner.model, baseline_model, cfg.device, encoder,
                num_episodes=cfg.eval_episodes
            )
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
