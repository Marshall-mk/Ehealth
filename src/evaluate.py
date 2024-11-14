import hydra
import mlflow
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from omegaconf import OmegaConf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from data import SkinLesionDataModule
from models import mobilenet_v2, mobilenet_v3
from torch.nn.functional import softmax
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.output_dir = Path("../evaluation_results")
        self.output_dir.mkdir(exist_ok=True)

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            probs = softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save and log
        cm_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

    def plot_roc_curves(self, y_true, y_prob):
        plt.figure(figsize=(10, 8))
        
        # One-hot encode true labels
        y_true_onehot = np.eye(len(self.class_names))[y_true]
        
        # Calculate ROC curve and ROC area for each class
        for i in range(len(self.class_names)):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, 
                tpr, 
                label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})'
            )

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        # Save and log
        roc_path = self.output_dir / "roc_curves.png"
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path)
        plt.close()

    def plot_precision_recall_curves(self, y_true, y_prob):
        plt.figure(figsize=(10, 8))
        
        # One-hot encode true labels
        y_true_onehot = np.eye(len(self.class_names))[y_true]
        
        # Calculate Precision-Recall curve and average precision for each class
        for i in range(len(self.class_names)):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_prob[:, i])
            avg_precision = average_precision_score(y_true_onehot[:, i], y_prob[:, i])
            plt.plot(
                recall, 
                precision, 
                label=f'{self.class_names[i]} (AP = {avg_precision:.2f})'
            )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="lower left")
        plt.tight_layout()
        
        # Save and log
        pr_path = self.output_dir / "precision_recall_curves.png"
        plt.savefig(pr_path)
        mlflow.log_artifact(pr_path)
        plt.close()

    def log_classification_report(self, y_true, y_pred):
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # Log metrics for each class
        for class_name in self.class_names:
            mlflow.log_metrics({
                f"{class_name}_precision": report[class_name]['precision'],
                f"{class_name}_recall": report[class_name]['recall'],
                f"{class_name}_f1-score": report[class_name]['f1-score'],
                f"{class_name}_support": report[class_name]['support']
            })
        
        # Log macro and weighted averages
        for avg_type in ['macro avg', 'weighted avg']:
            mlflow.log_metrics({
                f"{avg_type.replace(' ', '_')}_precision": report[avg_type]['precision'],
                f"{avg_type.replace(' ', '_')}_recall": report[avg_type]['recall'],
                f"{avg_type.replace(' ', '_')}_f1-score": report[avg_type]['f1-score']
            })

        # Save full report as text file
        report_path = self.output_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(classification_report(y_true, y_pred, target_names=self.class_names))
        mlflow.log_artifact(report_path)

@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    # Set up MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(f"{cfg.mlflow.experiment_name}_evaluation")
    
    with mlflow.start_run():
        # Log configuration
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        # Set device
        device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
        
        # Initialize data module and get test dataloader
        data_module = SkinLesionDataModule(cfg)
        data_module.setup()
        test_loader = data_module.get_test_dataloader()  # You'll need to add this method to your DataModule
        
        # Load model
        model_classes = {
            "mobilenetv2": mobilenet_v2,
            "mobilenetv3": mobilenet_v3
        }
        
        model_class = model_classes.get(cfg.model.model_name)
        if model_class is None:
            raise ValueError(f"Unknown model: {cfg.model.model_name}")
        
        model = model_class(output=len(data_module.class_names))
        model.load_state_dict(torch.load(cfg.evaluation.model_path))
        model = model.to(device)
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model=model,
            device=device,
            class_names=data_module.class_names
        )
        
        # Evaluate model
        print("Starting evaluation...")
        labels, predictions, probabilities = evaluator.evaluate(test_loader)
        
        # Generate and log evaluation metrics and plots
        evaluator.plot_confusion_matrix(labels, predictions)
        evaluator.plot_roc_curves(labels, probabilities)
        evaluator.plot_precision_recall_curves(labels, probabilities)
        evaluator.log_classification_report(labels, predictions)
        
        print("Evaluation completed. Results have been logged to MLflow.")

if __name__ == "__main__":
    main()