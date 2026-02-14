"""
Profile worker rollout to identify bottlenecks.

Runs a standalone worker (no Ray) and measures per-step timings for:
  A. env.step     — Rust environment stepping
  B. encode       — obs.encode() + obs.mask() + numpy->torch
  C. to_device    — tensor .to(device) transfer
  D. forward      — model forward pass (+ CUDA sync if GPU)
  E. action_select — mask apply + action selection
  F. store        — .cpu().numpy() + dict append

Usage:
    # CPU rollout
    uv run python scripts/profile_rollout.py -c configs/baseline.yml --device cpu

    # GPU rollout
    uv run python scripts/profile_rollout.py -c configs/baseline.yml --device cuda

    # GPU + torch.compile
    uv run python scripts/profile_rollout.py -c configs/baseline.yml --device cuda --compile

    # GPU + float16
    uv run python scripts/profile_rollout.py -c configs/baseline.yml --device cuda --half

    # GPU + batched multi-env rollout (4 envs)
    uv run python scripts/profile_rollout.py -c configs/baseline.yml --device cuda --batch_envs 4

    # Custom episodes / model
    uv run python scripts/profile_rollout.py -c configs/baseline.yml --device cuda --episodes 5 --load_model model.pth
"""
import argparse
import time

import numpy as np
import torch
from riichienv import RiichiEnv

from riichienv_ml.config import load_config, import_class
from riichienv_ml.training.ray_actor import sample_top_p


class TimingAccumulator:
    """Accumulates per-category timings across an episode."""

    CATEGORIES = ["env_step", "encode", "to_device", "forward", "action_select", "store"]

    def __init__(self):
        self.totals = {c: 0.0 for c in self.CATEGORIES}
        self.counts = {c: 0 for c in self.CATEGORIES}

    def add(self, category: str, elapsed: float):
        self.totals[category] += elapsed
        self.counts[category] += 1

    def report(self, episode_time: float, num_transitions: int) -> dict:
        """Returns a dict of {category: {total_s, count, mean_ms, pct}}."""
        results = {}
        for c in self.CATEGORIES:
            total = self.totals[c]
            count = self.counts[c]
            results[c] = {
                "total_s": total,
                "count": count,
                "mean_ms": (total / count * 1000) if count > 0 else 0.0,
                "pct": (total / episode_time * 100) if episode_time > 0 else 0.0,
            }
        measured = sum(self.totals.values())
        results["_overhead"] = {
            "total_s": episode_time - measured,
            "count": 0,
            "mean_ms": 0.0,
            "pct": ((episode_time - measured) / episode_time * 100) if episode_time > 0 else 0.0,
        }
        return results


def profile_episode(
    env, model, encoder, device,
    exploration="boltzmann",
    boltzmann_epsilon=0.02,
    boltzmann_temp=0.1,
    top_p=0.9,
    epsilon=0.1,
    input_dtype=torch.float32,
):
    """Run one episode and return detailed timing breakdown."""
    timer = TimingAccumulator()
    cuda = device.type == "cuda"

    # --- Reset ---
    t0 = time.perf_counter()
    obs_dict = env.reset(
        scores=[25000, 25000, 25000, 25000],
        round_wind=0, oya=0, honba=0, kyotaku=0,
    )
    t1 = time.perf_counter()
    timer.add("env_step", t1 - t0)

    episode_buffer = {0: [], 1: [], 2: [], 3: []}

    while not env.done():
        steps = {}
        for pid, obs in obs_dict.items():
            legal_actions = obs.legal_actions()
            if not legal_actions:
                continue

            # B. Encode
            t_enc0 = time.perf_counter()
            feat_tensor = encoder.encode(obs)
            mask = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
            mask_tensor = torch.from_numpy(mask)
            t_enc1 = time.perf_counter()
            timer.add("encode", t_enc1 - t_enc0)

            # C. To device
            t_dev0 = time.perf_counter()
            feat_tensor = feat_tensor.to(device=device, dtype=input_dtype)
            mask_tensor = mask_tensor.to(device)
            if cuda:
                torch.cuda.synchronize()
            t_dev1 = time.perf_counter()
            timer.add("to_device", t_dev1 - t_dev0)

            # D. Forward
            t_fwd0 = time.perf_counter()
            with torch.no_grad():
                feat_batch = feat_tensor.unsqueeze(0)
                q_values = model(feat_batch)
                mask_bool = mask_tensor.unsqueeze(0).bool()
                q_values = q_values.masked_fill(~mask_bool, -torch.inf)
            if cuda:
                torch.cuda.synchronize()
            t_fwd1 = time.perf_counter()
            timer.add("forward", t_fwd1 - t_fwd0)

            # E. Action select
            t_act0 = time.perf_counter()
            if exploration == "boltzmann":
                if np.random.random() < boltzmann_epsilon:
                    logits = q_values.float() / boltzmann_temp
                    logits = logits.masked_fill(~mask_bool, -torch.inf)
                    action_idx = sample_top_p(logits.squeeze(0), top_p).item()
                else:
                    action_idx = q_values.argmax(dim=1).item()
            else:
                if np.random.random() < epsilon:
                    legal_indices = [i for i, m in enumerate(mask_tensor.cpu().numpy()) if m > 0]
                    action_idx = np.random.choice(legal_indices)
                else:
                    action_idx = q_values.argmax(dim=1).item()
            t_act1 = time.perf_counter()
            timer.add("action_select", t_act1 - t_act0)

            found_action = obs.find_action(action_idx)
            if found_action is None:
                found_action = legal_actions[0]
            steps[pid] = found_action

            # F. Store transition
            t_st0 = time.perf_counter()
            episode_buffer[pid].append({
                "features": feat_tensor.float().cpu().numpy(),
                "mask": mask_tensor.cpu().numpy(),
                "action": action_idx,
            })
            t_st1 = time.perf_counter()
            timer.add("store", t_st1 - t_st0)

        # A. Env step
        t_env0 = time.perf_counter()
        obs_dict = env.step(steps)
        t_env1 = time.perf_counter()
        timer.add("env_step", t_env1 - t_env0)

    # Count transitions
    num_transitions = sum(len(v) for v in episode_buffer.values())
    return timer, num_transitions


def profile_batched_episodes(
    num_envs, model, encoder, device,
    exploration="boltzmann",
    boltzmann_epsilon=0.02,
    boltzmann_temp=0.1,
    top_p=0.9,
    epsilon=0.1,
    input_dtype=torch.float32,
):
    """Run multiple envs in parallel with batched inference. Returns timing and total transitions."""
    timer = TimingAccumulator()
    cuda = device.type == "cuda"

    envs = [RiichiEnv(game_mode="4p-red-half") for _ in range(num_envs)]

    # Reset all
    t0 = time.perf_counter()
    obs_dicts = [
        env.reset(scores=[25000]*4, round_wind=0, oya=0, honba=0, kyotaku=0)
        for env in envs
    ]
    t1 = time.perf_counter()
    timer.add("env_step", t1 - t0)

    active = [True] * num_envs
    all_buffers = [{0: [], 1: [], 2: [], 3: []} for _ in range(num_envs)]

    while any(active):
        # Collect observations from all active envs
        t_enc0 = time.perf_counter()
        batch_items = []  # (env_idx, pid, obs, legal_actions)
        for ei in range(num_envs):
            if not active[ei]:
                continue
            for pid, obs in obs_dicts[ei].items():
                la = obs.legal_actions()
                if la:
                    batch_items.append((ei, pid, obs, la))

        if not batch_items:
            # Check if all envs done
            for ei in range(num_envs):
                if active[ei] and envs[ei].done():
                    active[ei] = False
            if not any(active):
                break
            # Deadlock fallback
            break

        feat_list = []
        mask_list = []
        for ei, pid, obs, la in batch_items:
            feat_list.append(encoder.encode(obs))
            m = np.frombuffer(obs.mask(), dtype=np.uint8).copy()
            mask_list.append(torch.from_numpy(m))
        t_enc1 = time.perf_counter()
        timer.add("encode", t_enc1 - t_enc0)

        # To device (batched)
        t_dev0 = time.perf_counter()
        feat_batch = torch.stack(feat_list).to(device=device, dtype=input_dtype)
        mask_batch = torch.stack(mask_list).to(device)
        if cuda:
            torch.cuda.synchronize()
        t_dev1 = time.perf_counter()
        timer.add("to_device", t_dev1 - t_dev0)

        # Batched forward
        t_fwd0 = time.perf_counter()
        with torch.no_grad():
            q_values = model(feat_batch)
            mask_bool = mask_batch.bool()
            q_values = q_values.masked_fill(~mask_bool, -torch.inf)
        if cuda:
            torch.cuda.synchronize()
        t_fwd1 = time.perf_counter()
        timer.add("forward", t_fwd1 - t_fwd0)

        # Action selection (per item, but on GPU tensors)
        t_act0 = time.perf_counter()
        actions = q_values.argmax(dim=1)  # greedy default
        if exploration == "boltzmann" and boltzmann_epsilon > 0:
            # Apply boltzmann for some fraction
            logits = q_values.float() / boltzmann_temp
            logits = logits.masked_fill(~mask_bool, -torch.inf)
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)
            use_boltzmann = torch.rand(len(batch_items), device=device) < boltzmann_epsilon
            actions = torch.where(use_boltzmann, sampled, actions)
        actions_cpu = actions.cpu().numpy()
        t_act1 = time.perf_counter()
        timer.add("action_select", t_act1 - t_act0)

        # Store + build steps
        t_st0 = time.perf_counter()
        feat_cpu = feat_batch.float().cpu().numpy()
        mask_cpu = mask_batch.cpu().numpy()

        env_steps = {ei: {} for ei in range(num_envs)}
        for idx, (ei, pid, obs, la) in enumerate(batch_items):
            action_idx = int(actions_cpu[idx])
            all_buffers[ei][pid].append({
                "features": feat_cpu[idx],
                "mask": mask_cpu[idx],
                "action": action_idx,
            })
            found_action = obs.find_action(action_idx)
            if found_action is None:
                found_action = la[0]
            env_steps[ei][pid] = found_action
        t_st1 = time.perf_counter()
        timer.add("store", t_st1 - t_st0)

        # Step all active envs
        t_env0 = time.perf_counter()
        for ei in range(num_envs):
            if not active[ei]:
                continue
            if env_steps[ei]:
                obs_dicts[ei] = envs[ei].step(env_steps[ei])
            if envs[ei].done():
                active[ei] = False
        t_env1 = time.perf_counter()
        timer.add("env_step", t_env1 - t_env0)

    total_transitions = sum(
        sum(len(v) for v in buf.values()) for buf in all_buffers
    )
    return timer, total_transitions


def print_report(timers: list[TimingAccumulator], episode_times: list[float], transition_counts: list[int], device_name: str):
    """Print aggregated profiling report."""
    n = len(timers)

    # Aggregate
    agg_totals = {c: 0.0 for c in TimingAccumulator.CATEGORIES}
    agg_counts = {c: 0 for c in TimingAccumulator.CATEGORIES}
    total_time = sum(episode_times)
    total_transitions = sum(transition_counts)

    for t in timers:
        for c in TimingAccumulator.CATEGORIES:
            agg_totals[c] += t.totals[c]
            agg_counts[c] += t.counts[c]

    measured = sum(agg_totals.values())
    overhead = total_time - measured

    print(f"\n{'='*70}")
    print(f" Rollout Profile — device={device_name}, episodes={n}")
    print(f"{'='*70}")
    print(f"  Total time:        {total_time:.3f}s")
    print(f"  Total transitions: {total_transitions}")
    print(f"  Throughput:        {total_transitions/total_time:.1f} trans/s")
    print(f"  Avg episode:       {total_time/n:.3f}s, {total_transitions/n:.0f} trans")
    print()
    print(f"  {'Category':<16} {'Total(s)':>9} {'Count':>7} {'Mean(ms)':>10} {'%':>7}")
    print(f"  {'-'*16} {'-'*9} {'-'*7} {'-'*10} {'-'*7}")

    for c in TimingAccumulator.CATEGORIES:
        total_s = agg_totals[c]
        count = agg_counts[c]
        mean_ms = (total_s / count * 1000) if count > 0 else 0.0
        pct = (total_s / total_time * 100) if total_time > 0 else 0.0
        print(f"  {c:<16} {total_s:>9.3f} {count:>7d} {mean_ms:>10.3f} {pct:>6.1f}%")

    pct_overhead = (overhead / total_time * 100) if total_time > 0 else 0.0
    print(f"  {'overhead':<16} {overhead:>9.3f} {'':>7} {'':>10} {pct_overhead:>6.1f}%")
    print(f"{'='*70}")

    # Per-episode summary
    print(f"\n  Per-episode breakdown:")
    for i, (et, tc) in enumerate(zip(episode_times, transition_counts)):
        tps = tc / et if et > 0 else 0
        print(f"    Episode {i}: {et:.3f}s, {tc} trans, {tps:.1f} trans/s")
    print()


def main():
    parser = argparse.ArgumentParser(description="Profile worker rollout (standalone, no Ray)")
    parser.add_argument("-c", "--config", type=str, default="configs/baseline.yml")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to profile")
    parser.add_argument("--load_model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup episodes (not included in report)")
    parser.add_argument("--num_threads", type=int, default=1, help="torch.set_num_threads (default=1, same as worker)")
    # GPU optimizations
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile to model")
    parser.add_argument("--half", action="store_true", help="Use float16 inference")
    # Batched multi-env rollout
    parser.add_argument("--batch_envs", type=int, default=0, help="Number of envs for batched rollout (0=disabled)")
    # Model overrides
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--conv_channels", type=int, default=None)
    parser.add_argument("--fc_dim", type=int, default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)

    cfg = load_config(args.config).online
    model_config = cfg.model.model_dump()

    # Apply model overrides
    for field in ["num_blocks", "conv_channels", "fc_dim"]:
        val = getattr(args, field, None)
        if val is not None:
            model_config[field] = val

    device = torch.device(args.device)
    input_dtype = torch.float16 if args.half else torch.float32

    # Build label for report
    opts = []
    if args.compile:
        opts.append("compile")
    if args.half:
        opts.append("fp16")
    if args.batch_envs > 0:
        opts.append(f"batch{args.batch_envs}")
    device_label = args.device + ("+" + "+".join(opts) if opts else "")

    print(f"Device: {device} ({device_label})")
    print(f"Model config: {model_config}")
    print(f"Encoder: {cfg.encoder_class}")
    print(f"torch.num_threads: {torch.get_num_threads()}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name()}")

    # Build model
    ModelClass = import_class(cfg.model_class)
    model = ModelClass(**model_config).to(device)
    model.eval()

    # Load weights
    load_path = args.load_model or cfg.load_model
    if load_path:
        try:
            state = torch.load(load_path, map_location=device)
            model.load_state_dict(state, strict=False)
            print(f"Loaded weights from {load_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {load_path}: {e}")
            print("Continuing with random weights.")

    # Apply half precision
    if args.half:
        model = model.half()
        print("Model converted to float16")

    # Apply torch.compile
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled")

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    encoder = import_class(cfg.encoder_class)

    common_kwargs = dict(
        exploration=cfg.exploration,
        boltzmann_epsilon=cfg.boltzmann_epsilon,
        boltzmann_temp=cfg.boltzmann_temp_start,
        top_p=cfg.top_p,
        epsilon=cfg.epsilon_start,
        input_dtype=input_dtype,
    )

    if args.batch_envs > 0:
        # --- Batched multi-env mode ---
        print(f"\nBatched mode: {args.batch_envs} envs per batch")

        # Warmup
        if args.warmup > 0:
            print(f"Running {args.warmup} warmup round(s)...")
            for i in range(args.warmup):
                _, n = profile_batched_episodes(
                    args.batch_envs, model, encoder, device, **common_kwargs,
                )
                print(f"  Warmup {i}: {n} transitions")

        # Profile
        print(f"\nProfiling {args.episodes} round(s)...")
        timers = []
        round_times = []
        transition_counts = []

        for i in range(args.episodes):
            t0 = time.perf_counter()
            timer, n_trans = profile_batched_episodes(
                args.batch_envs, model, encoder, device, **common_kwargs,
            )
            t1 = time.perf_counter()
            rt = t1 - t0
            timers.append(timer)
            round_times.append(rt)
            transition_counts.append(n_trans)
            print(f"  Round {i}: {rt:.3f}s, {n_trans} transitions")

        print_report(timers, round_times, transition_counts, device_label)
    else:
        # --- Single-env mode ---
        env = RiichiEnv(game_mode="4p-red-half")

        # Warmup
        if args.warmup > 0:
            print(f"\nRunning {args.warmup} warmup episode(s)...")
            for i in range(args.warmup):
                _, n = profile_episode(
                    env, model, encoder, device, **common_kwargs,
                )
                print(f"  Warmup {i}: {n} transitions")

        # Profile
        print(f"\nProfiling {args.episodes} episode(s)...")
        timers = []
        episode_times = []
        transition_counts = []

        for i in range(args.episodes):
            t0 = time.perf_counter()
            timer, n_trans = profile_episode(
                env, model, encoder, device, **common_kwargs,
            )
            t1 = time.perf_counter()
            episode_time = t1 - t0
            timers.append(timer)
            episode_times.append(episode_time)
            transition_counts.append(n_trans)
            print(f"  Episode {i}: {episode_time:.3f}s, {n_trans} transitions")

        print_report(timers, episode_times, transition_counts, device_label)


if __name__ == "__main__":
    main()
