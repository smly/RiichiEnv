import sys
import os
import ray
import argparse

import wandb
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import torch
from torch.distributions import Categorical
from riichienv import RiichiEnv

from ray_actor import MahjongWorker
from learner import MahjongLearner
from buffer import GlobalReplayBuffer
from cql_dataset import ObservationEncoder


def evaluate_vs_baseline(hero_model, baseline_model, device, num_episodes=30):
    """
    Evaluates Hero Model (Player 0) vs Baseline Model (Players 1-3).
    Running on single GPU sequentially (batch size 1 per turn).
    Ideally this should be batched, but sequential is fine for 30 games.
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
                if pid == 0:
                    model = hero_model
                else:
                    model = baseline_model

                # Encode using legacy features (46, 34)
                feat = ObservationEncoder.encode_legacy(obs)
                mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()

                # Move to device and add batch dimension
                feat_t = feat.to(device).unsqueeze(0)
                mask_t = torch.from_numpy(mask).to(device).unsqueeze(0)

                with torch.no_grad():
                    logits, q_values = model(feat_t)

                    if pid == 0:
                        # Hero (PPO): Sample from Policy (Logits)
                        logits = logits.masked_fill(mask_t == 0, -1e9)
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                        action_idx = action.item()
                    else:
                        # Baseline (CQL): Greedy Argmax Q-values
                        q_values = q_values.masked_fill(mask_t == 0, -1e9)
                        action_idx = q_values.argmax(dim=1).item()

                found_action = obs.find_action(action_idx)
                if found_action is None:
                    found_action = obs.legal_actions()[0]

                steps[pid] = found_action

            obs_dict = env.step(steps)

        ranks = env.ranks()
        
        # Reward Config: [10, 4, -4, -10]
        # RiichiEnv returns 1-based ranks (1, 2, 3, 4)
        rank = ranks[0]
        reward = 0.0
        if rank == 1: reward = 10.0
        elif rank == 2: reward = 4.0
        elif rank == 3: reward = -4.0
        elif rank == 4: reward = -10.0

        hero_rewards.append(reward)
        hero_ranks.append(rank)
        
    return np.mean(hero_rewards), np.mean(hero_ranks)


def proper_loop(args):
    # Set working_dir to demos/ml_baseline (Current Directory)
    # Explicitly pass current sys.path to workers
    python_path = ":".join(sys.path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    runtime_env = {
        "working_dir": current_dir,
        "excludes": [".git", ".venv", "wandb", "__pycache__", "pyproject.toml", "uv.lock"],
        "env_vars": {
            "PYTHONPATH": python_path,
            "PATH": os.environ["PATH"]
        }
    }
    
    ray.init(runtime_env=runtime_env, ignore_reinit_error=True)

    learner = MahjongLearner(
        device=args.device,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_cql_init=args.alpha_cql_init,
        alpha_cql_final=args.alpha_cql_final,
        awac_beta=args.awac_beta,
        awac_max_weight=args.awac_max_weight,
    )
    baseline_learner = None

    if args.load_model:
        learner.load_cql_weights(args.load_model)
        # Load Baseline for Evaluation
        baseline_learner = MahjongLearner(device=args.device)
        baseline_learner.load_cql_weights(args.load_model)
        baseline_learner.model.eval() # Freeze baseline

    # AWAC is off-policy: use unified buffer (same capacity for actor and critic)
    buffer = GlobalReplayBuffer(
        batch_size=args.batch_size,
        device=args.device,
        actor_capacity=args.critic_capacity,  # Unified buffer
        critic_capacity=args.critic_capacity,
    )
    # Distribute workers across GPUs if available
    # Each worker gets a fraction of GPU memory
    if args.worker_device == "cuda":
        # Assign workers to GPU with memory fraction
        workers = [
            MahjongWorker.options(num_gpus=args.gpu_per_worker).remote(i, "cuda", gamma=0.99)
            for i in range(args.num_workers)
        ]
    else:
        workers = [MahjongWorker.remote(i, "cpu", gamma=0.99) for i in range(args.num_workers)]
    
    # Initial Weight Sync
    weights = {k: v.cpu() for k,v in learner.get_weights().items()}
    weight_ref = ray.put(weights)

    for w in workers:
        w.update_weights.remote(weight_ref)
        
    # Map future -> worker_index
    future_to_worker = {w.collect_episode.remote(): i for i, w in enumerate(workers)}
    
    step = 0
    wandb.init(project="riichienv-ppo-cql", config=vars(args))
    
    # Create checkpoints dir
    os.makedirs("checkpoints", exist_ok=True)
    
    # Dry Run Evaluation to verify stability
    if baseline_learner is not None:
        print("Running dry-run evaluation (1 episode)...")
        try:
            evaluate_vs_baseline(learner.model, baseline_learner.model, args.device, num_episodes=30)
            print("Dry-run evaluation passed.")
        except Exception as e:
            print(f"Dry-run evaluation failed: {e}")
            raise e

    try:
        while step < args.num_steps:
            # 1. Periodic Evaluation & Snapshot (Priority over Training Step)
            if step > 0 and step % args.eval_interval == 0:
                print(f"Step {step}: Saving snapshot and Evaluating...")
                sys.stdout.flush()

                # Snapshot
                save_path = f"checkpoints/model_{step}.pth"
                torch.save(learner.get_weights(), save_path)
                print(f"Saved snapshot to {save_path}")
                
                # Evaluaton
                if baseline_learner is not None:
                    try:
                        print("Starting Evaluation loop...")
                        eval_reward, eval_rank = evaluate_vs_baseline(
                            learner.model, baseline_learner.model, args.device, num_episodes=30
                        )
                        print(f"Evaluation (vs Baseline): Reward={eval_reward:.2f}, Rank={eval_rank:.2f}")
                        wandb.log({
                            "eval/reward": eval_reward, 
                            "eval/rank": eval_rank
                        }, step=step)
                    except Exception as e:
                        print(f"Evaluation failed at step {step}: {e}")

            ready_ids, _ = ray.wait(list(future_to_worker.keys()), num_returns=1)
            future = ready_ids[0]
            worker_idx = future_to_worker.pop(future)
            
            # Get Result
            transitions = ray.get(future)
            buffer.add(transitions)

            # Train - Single gradient step per episode (matching main branch performance)
            metrics = {}

            # Update Critic (using historical data)
            if len(buffer.critic_buffer) > args.batch_size:
                c_batch = buffer.sample_critic(args.batch_size)

                # Update with Priority Logic and dynamic CQL alpha
                # update_critic returns (metrics, indices, priorities)
                c_metrics, indices, priorities = learner.update_critic(c_batch, max_steps=args.num_steps)
                metrics.update(c_metrics)

                # Feedback priorities to buffer
                buffer.update_priority(indices, priorities)

            # Update Actor (AWAC is off-policy, can use same data as critic)
            if len(buffer.actor_buffer) > args.batch_size:
                a_batch = buffer.sample_actor(args.batch_size)
                a_metrics = learner.update_actor(a_batch)
                metrics.update(a_metrics)
                
            if step % 200 == 0 and step > 0:
                print(f"Step {step}: {metrics}")
                wandb.log(metrics, step=step)
            
            # Re-dispatch worker
            # Sync weights periodically to reduce overhead (every N steps)
            if step % args.weight_sync_freq == 0:
                weights = {k: v.cpu() for k,v in learner.get_weights().items()}

                # Check for NaN in weights before sending
                has_nan = False
                for name, param in weights.items():
                    if torch.isnan(param).any():
                        print(f"ERROR: NaN detected in learner weights {name} at step {step}")
                        has_nan = True

                if has_nan:
                    print(f"FATAL: Cannot sync NaN weights to workers. Stopping training.")
                    break

                weight_ref = ray.put(weights)
                # Update the worker that just finished
                workers[worker_idx].update_weights.remote(weight_ref)
            else:
                # Just re-dispatch without weight update for efficiency
                pass
            
            new_future = workers[worker_idx].collect_episode.remote()
            future_to_worker[new_future] = worker_idx
            
            step += 1
            
    except KeyboardInterrupt:
        print("Stopping...")
        
        ray.shutdown()
        
    # Final Snapshot and Evaluation
    print(f"Final Step {step}: Saving snapshot and Evaluating...")
    sys.stdout.flush()
    
    save_path = f"checkpoints/model_{step}.pth"
    torch.save(learner.get_weights(), save_path)
    print(f"Saved snapshot to {save_path}")
    
    if baseline_learner is not None:
        try:
            print("Starting Final Evaluation loop...")
            eval_reward, eval_rank = evaluate_vs_baseline(
                learner.model, baseline_learner.model, args.device, num_episodes=30
            )
            print(f"Final Evaluation (vs Baseline): Reward={eval_reward:.2f}, Rank={eval_rank:.2f}")
            wandb.log({
                "eval/reward": eval_reward, 
                "eval/rank": eval_rank
            }, step=step)
        except Exception as e:
            print(f"Final Evaluation failed: {e}")

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--num_steps", type=int, default=1e5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load_model", type=str, default=None, help="Path to offline CQL model (cql_model.pth)")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--critic_lr", type=float, default=3e-4, help="Critic learning rate")
    parser.add_argument("--alpha_cql_init", type=float, default=1.0, help="Initial CQL alpha")
    parser.add_argument("--alpha_cql_final", type=float, default=0.1, help="Final CQL alpha")
    parser.add_argument("--awac_beta", type=float, default=0.3, help="AWAC temperature (lower = more conservative)")
    parser.add_argument("--awac_max_weight", type=float, default=20.0, help="AWAC max advantage weight clipping")
    parser.add_argument("--actor_capacity", type=int, default=50000, help="Actor buffer capacity (unused with AWAC unified buffer)")
    parser.add_argument("--critic_capacity", type=int, default=1000000, help="Critic buffer capacity")
    parser.add_argument("--eval_interval", type=int, default=2000, help="Evaluation interval")
    parser.add_argument("--weight_sync_freq", type=int, default=10, help="Sync weights to workers every N steps")
    parser.add_argument("--worker_device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for workers (cpu or cuda)")
    parser.add_argument("--gpu_per_worker", type=float, default=0.1, help="Fraction of GPU memory per worker (e.g., 0.1 = 10%)")
    args = parser.parse_args()

    proper_loop(args)
