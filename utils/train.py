from dataclasses import dataclass

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

@dataclass
class TrainConfig:
    # Training parameters
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    patience: int = 5
    min_delta: float = 0.01
    warmup_steps: int = 5000
    log_interval: int = 1000

def initialize_training_components(model: nn.Module, train_dataset, device: str, config: TrainConfig):
    scaler = GradScaler()
    pos_weight = torch.tensor([len(train_dataset[train_dataset['target'] == 0]) / 
                               len(train_dataset[train_dataset['target'] == 1])]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    