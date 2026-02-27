"""Inference and evaluation for hand prediction model.

Subcommands:
    eval     Evaluate on replay data (prediction vs ground truth metrics)
    predict  Predict opponent hands from a single Observation (base64)

Usage:
    uv run python scripts/infer_hand_pred.py eval \
        -c src/riichienv_ml/configs/4p/hand_pred_cnn.yml \
        --model_path /path/to/hand_pred_cnn.pth \
        --data_glob "/data/mjsoul/mjsoul-4p/2024/01/**/*.jsonl.gz"

    uv run python scripts/infer_hand_pred.py predict \
        -c src/riichienv_ml/configs/4p/hand_pred_cnn.yml \
        --model_path /path/to/hand_pred_cnn.pth \
        --obs_b64 "<base64-encoded Observation>"
"""

import argparse
import json
import sys

import numpy as np
import torch
import tqdm
from loguru import logger
from torch.utils.data import DataLoader

from riichienv import Observation
from riichienv_ml.config import HandPredConfig, import_class, load_config
from riichienv_ml.datasets.hand_pred import HandPredDataset
from riichienv_ml.utils import AverageMeter


def _load_model(cfg: HandPredConfig, model_path: str, device: torch.device):
    ModelCls = import_class(cfg.model_class)
    model_kwargs = cfg.model.model_dump()
    model_kwargs["tile_dim"] = cfg.game.tile_dim
    model = ModelCls(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def _load_encoder(cfg: HandPredConfig):
    EncoderCls = import_class(cfg.encoder_class)
    return EncoderCls(tile_dim=cfg.game.tile_dim)


# ── Tile names for display ──────────────────────────────────────────
_TILE_NAMES_34 = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
]


# ── eval subcommand ─────────────────────────────────────────────────
def cmd_eval(args):
    cfg = load_config(args.config).hand_pred
    if args.data_glob:
        cfg = cfg.model_copy(update={"val_data_glob": args.data_glob})
    device = torch.device(cfg.device)

    model = _load_model(cfg, args.model_path, device)
    encoder = _load_encoder(cfg)

    dataset = HandPredDataset(
        data_glob=cfg.val_data_glob,
        n_players=cfg.game.n_players,
        tile_dim=cfg.game.tile_dim,
        replay_rule=cfg.game.replay_rule,
        is_train=False,
        encoder=encoder,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    loss_meter = AverageMeter("loss", ":.4e")
    mse_meter = AverageMeter("mse", ":.4e")
    mae_meter = AverageMeter("mae", ":.4f")
    tile_acc_meter = AverageMeter("tile_acc", ":.4f")
    sum_err_meter = AverageMeter("sum_err", ":.4f")

    # Per-phase meters (early/mid/late based on step index within buffer)
    phase_mae = {"early": AverageMeter("e", ":.4f"),
                 "mid": AverageMeter("m", ":.4f"),
                 "late": AverageMeter("l", ":.4f")}
    phase_acc = {"early": AverageMeter("e", ":.4f"),
                 "mid": AverageMeter("m", ":.4f"),
                 "late": AverageMeter("l", ":.4f")}

    step_count = 0
    criterion = torch.nn.SmoothL1Loss()

    pbar = tqdm.tqdm(dataloader, desc="eval", mininterval=1.0, ncols=120)
    with torch.no_grad():
        for x, y, cc in pbar:
            x, y, cc = x.to(device), y.to(device), cc.to(device)
            pred = model(x)
            bs = x.size(0)

            loss_meter.update(criterion(pred, y).item(), bs)
            mse_meter.update(((pred - y) ** 2).mean().item(), bs)
            mae_val = (pred - y).abs().mean().item()
            mae_meter.update(mae_val, bs)
            acc_val = (pred.round() == y).float().mean().item()
            tile_acc_meter.update(acc_val, bs)
            # Sum error: |predicted total - actual concealed count|
            sum_err = (pred.sum(dim=-1) - cc).abs().mean().item()
            sum_err_meter.update(sum_err, bs)

            # Approximate game phase from global step count
            for i in range(bs):
                local_step = step_count + i
                phase_idx = local_step % 60  # rough heuristic
                if phase_idx < 20:
                    phase = "early"
                elif phase_idx < 40:
                    phase = "mid"
                else:
                    phase = "late"
                sample_mae = (pred[i] - y[i]).abs().mean().item()
                sample_acc = (pred[i].round() == y[i]).float().mean().item()
                phase_mae[phase].update(sample_mae, 1)
                phase_acc[phase].update(sample_acc, 1)
            step_count += bs

            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4e}",
                mae=f"{mae_meter.avg:.4f}",
                acc=f"{tile_acc_meter.avg:.4f}",
            )

    results = {
        "total_samples": loss_meter.count,
        "smooth_l1_loss": loss_meter.avg,
        "mse": mse_meter.avg,
        "mae": mae_meter.avg,
        "tile_accuracy": tile_acc_meter.avg,
        "sum_error": sum_err_meter.avg,
    }
    for phase in ("early", "mid", "late"):
        results[f"mae_{phase}"] = phase_mae[phase].avg if phase_mae[phase].count > 0 else None
        results[f"tile_acc_{phase}"] = phase_acc[phase].avg if phase_acc[phase].count > 0 else None

    logger.info("Evaluation results:")
    for k, v in results.items():
        if v is not None:
            logger.info(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print(json.dumps(results, indent=2))


# ── predict subcommand ───────────────────────────────────────────────
def cmd_predict(args):
    cfg = load_config(args.config).hand_pred
    device = torch.device(cfg.device)

    model = _load_model(cfg, args.model_path, device)
    encoder = _load_encoder(cfg)

    obs = Observation.deserialize_from_base64(args.obs_b64)
    features = encoder.encode(obs).unsqueeze(0).to(device)  # (1, ...)

    with torch.no_grad():
        pred = model(features)  # (1, 3, tile_dim)

    pred_np = pred.squeeze(0).cpu().numpy()  # (3, tile_dim)
    tile_dim = cfg.game.tile_dim
    n_players = cfg.game.n_players
    tile_names = _TILE_NAMES_34[:tile_dim]

    result = {
        "player_id": obs.player_id,
        "opponents": [],
    }
    for rel in range(1, n_players):
        abs_id = (obs.player_id + rel) % n_players
        opp_pred = pred_np[rel - 1]
        tiles = {}
        for t in range(tile_dim):
            if opp_pred[t] > 0.3:  # threshold for display
                tiles[tile_names[t]] = round(float(opp_pred[t]), 2)
        result["opponents"].append({
            "seat": abs_id,
            "relation": ["shimocha", "toimen", "kamicha"][rel - 1],
            "predicted_tiles": tiles,
            "predicted_total": round(float(opp_pred.sum()), 1),
            "raw_counts": [round(float(v), 3) for v in opp_pred],
        })

    print(json.dumps(result, indent=2, ensure_ascii=False))


# ── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hand prediction inference / evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate on replay data")
    p_eval.add_argument("-c", "--config", type=str, required=True)
    p_eval.add_argument("--model_path", type=str, required=True)
    p_eval.add_argument("--data_glob", type=str, default=None,
                        help="Override val_data_glob from config")
    p_eval.set_defaults(func=cmd_eval)

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict from base64 Observation")
    p_pred.add_argument("-c", "--config", type=str, required=True)
    p_pred.add_argument("--model_path", type=str, required=True)
    p_pred.add_argument("--obs_b64", type=str, required=True,
                        help="Base64-encoded Observation")
    p_pred.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
