import os
import random

import torch
from omegaconf import DictConfig
import hydra

from src.utils.data_utils import get_dataloaders
from src.utils.model_utils import build_tokenizer_and_model, get_optim_and_scheduler
from src.utils.train_utils import train_epoch, validate_epoch

@hydra.main(config_path="configs", config_name=None, version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer, model = build_tokenizer_and_model(cfg)
    train_dl, val_dl = get_dataloaders(cfg, tokenizer)
    total_steps = cfg.training.epochs * len(train_dl)
    optim, scheduler = get_optim_and_scheduler(cfg, model, total_steps)

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = train_epoch(model, train_dl, optim, scheduler, None, None, cfg, epoch)
        val_loss = validate_epoch(model, val_dl, cfg, epoch)

    torch.save(model.state_dict(), "model.pt")
    tokenizer.save_pretrained("tokenizer")
    print("model saved")

if __name__ == "__main__":
    main()

