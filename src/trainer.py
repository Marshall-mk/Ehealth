import torch
import time
import copy
import os
import mlflow
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from utils import EarlyStopper

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, cfg):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = cfg
        self.early_stopping = EarlyStopper(patience=cfg.train.patience, min_delta=10)
        
    def train_model(self, dataloaders, dataset_sizes, class_names):
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_params({
            "model_name": self.cfg.model.model_name,
            "batch_size": self.cfg.train.batch_size,
            "epochs": self.cfg.train.epochs,
            "loss": self.cfg.train.loss,
            "optimizer": self.cfg.train.optimizer,
            "learning_rate": self.cfg.train.learning_rate,
            "scheduler": self.cfg.train.scheduler,
            "patience": self.cfg.train.patience,
            "device": self.cfg.train.device,
            
        })
        
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        best_loss = float('inf')
        best_epoch = 0

        for epoch in range(self.cfg.train.epochs):
            print(f'Epoch {epoch+1}/{self.cfg.train.epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                metrics = self._train_epoch(phase, dataloaders[phase], dataset_sizes[phase], class_names, epoch)
                
                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{phase}_{metric_name}", metric_value, step=epoch)
                
                if phase == 'val':
                    if metrics['accuracy'] > best_acc or metrics['loss'] < best_loss:
                        best_acc = metrics['accuracy']
                        best_loss = metrics['loss']
                        best_model_wts = copy.deepcopy(self.model.state_dict())
                        self._save_checkpoint(epoch)
                        best_epoch = epoch + 1
                        mlflow.log_metric("best_epoch", best_epoch)
                    
                    if self.early_stopping.early_stop(metrics['loss']):
                        print("Early stopping triggered")
                        mlflow.log_metric("early_stopped_epoch", epoch)
                        break

            if self.early_stopping.early_stop(metrics['loss']):
                break

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        print(f'Best val Loss: {best_loss:.4f}')
        print(f'Best epoch: {best_epoch}')
        mlflow.log_metric("best_epoch", best_epoch)
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        mlflow.end_run()
        return self.model

    def _train_epoch(self, phase, dataloader, dataset_size, class_names, epoch):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        all_preds = []
        all_labels = []

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size
        
        metrics = self._compute_metrics(all_labels, all_preds, epoch_loss, epoch_acc, class_names)
        
        if phase == 'val':
            self._update_scheduler(epoch_loss)
            
        return metrics

    def _compute_metrics(self, all_labels, all_preds, epoch_loss, epoch_acc, class_names):
        cm = confusion_matrix(all_labels, all_preds)
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        precision = np.mean([cm[i,i] / np.sum(cm[:,i]) for i in range(len(class_names))])
        recall = np.mean([cm[i,i] / np.sum(cm[i,:]) for i in range(len(class_names))])
        f1_score = 2 * (precision * recall) / (precision + recall)
        kappa = cohen_kappa_score(all_labels, all_preds)
        
        return {
            'loss': epoch_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'kappa': kappa
        }

    def _save_checkpoint(self, epoch):
        checkpoint_dir = "../checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = f"{checkpoint_dir}/{self.cfg.model.model_name}_epoch_{epoch+1}.pth"
        torch.save(self.model.state_dict(), path)
        mlflow.log_artifact(path)

    def _update_scheduler(self, epoch_loss):
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_loss)
            else:
                self.scheduler.step()