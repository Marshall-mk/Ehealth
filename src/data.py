import torch
from torchvision import datasets, transforms

class SkinLesionDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mean = [0.6104, 0.5033, 0.4965]
        self.std = [0.2507, 0.2288, 0.2383]
        self.data_transforms = self._create_transforms()
        
    def _create_transforms(self):
        return {
            "train": transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(self.cfg.model.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.cfg.model.image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        }
    
    def setup(self):
        self.train_dataset = datasets.ImageFolder(
            root=f"../{self.cfg.model.train_data_path}", 
            transform=self.data_transforms["train"]
        )
        self.val_dataset = datasets.ImageFolder(
            root=f"../{self.cfg.model.val_data_path}", 
            transform=self.data_transforms["val"]
        )
        
        self.class_names = self.train_dataset.classes
        self.dataset_sizes = {
            "train": len(self.train_dataset), 
            "val": len(self.val_dataset)
        }
        
        self.class_weights = self._calculate_class_weights()  # Moved here

    def _calculate_class_weights(self):  
        num_classes = self.cfg.model.num_classes
        weights = [1.0 / len(self.train_dataset) for _ in range(num_classes)]
        return torch.tensor(weights)
        
    def get_dataloaders(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=10
        )
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=10
        )
        return train_loader, val_loader
    
    def get_test_dataloader(self):
        test_dataset = datasets.ImageFolder(
            root=f"../{self.cfg.model.test_data_path}",
            transform=self.data_transforms["val"]  
        )
    
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            num_workers=10
                            )