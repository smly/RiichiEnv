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

                # Encode
                feat = np.frombuffer(obs.encode(), dtype=np.float32).reshape(46, 34).copy()
                mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()

                feat_t = torch.from_numpy(feat).to(device).unsqueeze(0)
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
    
    learner = MahjongLearner(device=args.device)
    baseline_learner = None
    
    if args.load_model:
        learner.load_cql_weights(args.load_model)
        # Load Baseline for Evaluation
        baseline_learner = MahjongLearner(device=args.device)
        baseline_learner.load_cql_weights(args.load_model)
        baseline_learner.model.eval() # Freeze baseline

    buffer = GlobalReplayBuffer(batch_size=args.batch_size, device=args.device)
    workers = [MahjongWorker.remote(i, "cpu") for i in range(args.num_workers)]
    
    # Initial Weight Sync
    weights = {k: v.cpu() for k,v in learner.get_weights().items()}
    weight_ref = ray.put(weights)
    
    for w in workers:
        w.update_weights.remote(weight_ref, 0)
        
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
    
    start_beta = 0.4
    end_beta = 0.8
    
    try:
        while step < args.num_steps:
            # 1. Periodic Evaluation & Snapshot (Priority over Training Step)
            if step > 0 and step % 10000 == 0:
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
            
            # Anneal Beta
            # Linear annealing from start_beta to end_beta over num_steps
            progress = min(1.0, step / args.num_steps)
            beta = start_beta + progress * (end_beta - start_beta)
            buffer.update_beta(beta)
            
            # Train
            metrics = {"beta": beta}
            
            # Update Critic (using historical data) -> Multiple steps?
            if len(buffer.critic_buffer) > args.batch_size:
                c_batch = buffer.sample_critic(args.batch_size)
                
                # Update with Priority Logic
                # update_critic returns (metrics, indices, priorities)
                c_metrics, indices, priorities = learner.update_critic(c_batch)
                metrics.update(c_metrics)
                
                # Feedback priorities to buffer
                buffer.update_priority(indices, priorities)
                
            # Update Actor (using recent data)
            if len(buffer.actor_buffer) > args.batch_size:
                a_batch = buffer.sample_actor(args.batch_size)
                a_metrics = learner.update_actor(a_batch)
                metrics.update(a_metrics)
                
            if step % 200 == 0 and step > 0:
                print(f"Step {step}: {metrics}")
                wandb.log(metrics, step=step)
            
            # Re-dispatch worker with latest weights
            # Sync weights every X steps or every episode?
            # Sync every time for now for On-Policy-ness
            weights = {k: v.cpu() for k,v in learner.get_weights().items()}
            weight_ref = ray.put(weights)
            workers[worker_idx].update_weights.remote(weight_ref, learner.policy_version)
            
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=1e5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load_model", type=str, default=None, help="Path to offline CQL model (cql_model.pth)")
    args = parser.parse_args()
    
    proper_loop(args)
