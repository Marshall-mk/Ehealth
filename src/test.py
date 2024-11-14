import hydra
import mlflow
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
from models import mobilenet_v1, mobilenet_v2, mobilenet_v3

class ImagePredictor:
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),  # Adjust based on your model's input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.6104, 0.5033, 0.4965], std=[0.2507, 0.2288, 0.2383]),
        ])

    def predict(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)  # Add batch dimension

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)

        return predicted_class.item(), confidence.item(), probabilities.cpu().numpy()

@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    # Set up MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        # Log configuration
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # Set device
        device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

        # Load model
        model_classes = {
            "mobilenet": mobilenet_v1,
            "mobilenetv2": mobilenet_v2,
            "mobilenetv3": mobilenet_v3
        }

        model_class = model_classes.get(cfg.model.model_name)
        if model_class is None:
            raise ValueError(f"Unknown model: {cfg.model.model_name}")

        model = model_class(output=len(cfg.model.class_names))
        model.load_state_dict(torch.load(cfg.evaluation.model_path))
        model = model.to(device)

        # Initialize predictor
        predictor = ImagePredictor(model=model, device=device, class_names=cfg.model.class_names)

        # Predict on the provided image
        image_path = cfg.test.image_path  # Path to the image to be tested
        predicted_class, confidence, probabilities = predictor.predict(image_path)

        # Log results
        mlflow.log_metric("predicted_class", predicted_class)
        mlflow.log_metric("confidence", confidence)
        mlflow.log_artifact(image_path)  # Log the image used for prediction

        # Print results
        print(f"Predicted Class: {predictor.class_names[predicted_class]}")
        print(f"Confidence: {confidence:.4f}")
        print("Class Probabilities:")
        for i, class_name in enumerate(predictor.class_names):
            print(f"{class_name}: {probabilities[0][i]:.4f}")

if __name__ == "__main__":
    main()