from __future__ import annotations

import importlib
from typing import Literal

import yaml
from pydantic import BaseModel


def import_class(dotted_path: str):
    """
    Dynamically import a class from a dotted path.
    e.g. "riichienv_ml.models.cql_model.QNetwork" -> QNetwork class
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class GrpConfig(BaseModel):
    train_data: list[str] = ["/data/train_grp.pq", "/data/train_grp_2024.pq"]
    val_data: str = "/data/val_grp.pq"
    output: str = "grp_model.pth"
    device: str = "cuda"
    batch_size: int = 128
    num_workers: int = 12
    num_epochs: int = 10
    lr: float = 5e-4
    input_dim: int = 20


class ModelConfig(BaseModel):
    in_channels: int = 74
    num_blocks: int = 3
    conv_channels: int = 64
    fc_dim: int = 256
    num_actions: int = 82


class CqlConfig(BaseModel):
    data_glob: str = "/data/mjsoul/mahjong_game_record_4p_thr_202[45]*/*.bin.xz"
    grp_model: str = "./grp_model.pth"
    output: str = "cql_model.pth"
    device: str = "cuda"
    batch_size: int = 32
    lr: float = 1e-4
    alpha: float = 1.0
    gamma: float = 0.99
    num_epochs: int = 10
    num_workers: int = 12
    limit: int = 3000000
    pts_weight: list[float] = [10.0, 4.0, -4.0, -10.0]
    wandb_entity: str = "smly"
    wandb_project: str = "riichienv-offline"
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.cql_model.QNetwork"
    dataset_class: str = "riichienv_ml.data.cql_dataset.MCDataset"


class OnlineConfig(BaseModel):
    load_model: str | None = None
    device: str = "cuda"
    num_workers: int = 12
    num_steps: int = 5000000
    batch_size: int = 128
    lr: float = 1e-4
    alpha_cql_init: float = 1.0
    alpha_cql_final: float = 0.1
    # Exploration strategy: "epsilon_greedy" or "boltzmann"
    exploration: Literal["epsilon_greedy", "boltzmann"] = "boltzmann"
    # epsilon-greedy params
    epsilon_start: float = 0.1
    epsilon_final: float = 0.01
    # Boltzmann (softmax) exploration params
    boltzmann_epsilon: float = 0.02
    boltzmann_temp_start: float = 0.1
    boltzmann_temp_final: float = 0.05
    top_p: float = 1.0
    capacity: int = 1000000
    eval_interval: int = 2000
    weight_sync_freq: int = 10
    worker_device: Literal["cpu", "cuda"] = "cpu"
    gpu_per_worker: float = 0.1
    gamma: float = 0.99
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "riichienv-online"
    model: ModelConfig = ModelConfig()
    model_class: str = "riichienv_ml.models.cql_model.QNetwork"
    encoder_class: str = "riichienv_ml.data.cql_dataset.ObservationEncoder"


class Config(BaseModel):
    grp: GrpConfig = GrpConfig()
    cql: CqlConfig = CqlConfig()
    online: OnlineConfig = OnlineConfig()


def load_config(path: str) -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
