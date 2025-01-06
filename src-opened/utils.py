import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.ops import sigmoid_focal_loss 

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
def get_criterion(cfg, class_weights):
    if cfg.train.loss == "FocalLoss":
        return lambda inputs, targets: sigmoid_focal_loss(inputs, targets.float().unsqueeze(1).expand(-1, cfg.model.num_classes), alpha=0.25, gamma=2.0, reduction="mean")
    elif cfg.train.loss == "WeightedCELoss":
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    return torch.nn.CrossEntropyLoss()

def get_optimizer(cfg, model):
    optimizers = {
        "Adam": optim.Adam,
        "SGD": lambda p, lr: optim.SGD(p, lr=lr, momentum=0.9),
        "RMSprop": optim.RMSprop,
        "AdamW": optim.AdamW
    }
    optimizer_fn = optimizers.get(cfg.train.optimizer)
    if optimizer_fn is None:
        raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer}")
    return optimizer_fn(model.parameters(), lr=cfg.train.learning_rate)

def get_scheduler(cfg, optimizer):
    schedulers = {
        "StepLR": lambda: lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1),
        "MultiStepLR": lambda: lr_scheduler.MultiStepLR(optimizer, milestones=[7, 10], gamma=0.1),
        "ExponentialLR": lambda: lr_scheduler.ExponentialLR(optimizer, gamma=0.1),
        "ReduceLROnPlateau": lambda: lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5, verbose=True),
        "CosineAnnealingLR": lambda: lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0),
        "CosineAnnealingWarmRestarts": lambda: lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
    }
    scheduler_fn = schedulers.get(cfg.train.scheduler)
    if scheduler_fn is None:
        raise ValueError(f"Unsupported scheduler: {cfg.train.scheduler}")
    return scheduler_fn()