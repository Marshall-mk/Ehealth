import hydra
import mlflow
from omegaconf import OmegaConf
from data import SkinLesionDataModule
from trainer import ModelTrainer
from models import mobilenet_v2, mobilenet_v3
from utils import get_criterion, get_optimizer, get_scheduler
import warnings
warnings.filterwarnings("ignore")

@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg):
    OmegaConf.to_yaml(cfg, resolve=True)
    # Set up MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    # Initialize data module
    data_module = SkinLesionDataModule(cfg)
    data_module.setup()
    train_loader, val_loader = data_module.get_dataloaders()
    dataloaders = {"train": train_loader, "val": val_loader}
    
    # Train each model specified in config
    model_name = cfg.model.model_name
    print(f"Training {model_name}")
    
    # Model initialization
    model = {
        "mobilenetv2": mobilenet_v2,
        "mobilenetv3": mobilenet_v3
    }.get(model_name)(output=cfg.model.num_classes, ckpt=cfg.model.checkpoint)
    
    # Get training components
    criterion = get_criterion(cfg, data_module.class_weights)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg.train.device,
        cfg=cfg
    )
    
    # Train model
    trained_model = trainer.train_model(
        dataloaders=dataloaders,
        dataset_sizes=data_module.dataset_sizes,
        class_names=data_module.class_names
    )
    
    return trained_model

if __name__ == "__main__":
    main()