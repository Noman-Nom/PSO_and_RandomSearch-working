"""
Professional data loading utilities with train/val/test splits
"""
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np


class DatasetLoader:
    """Unified interface for loading datasets with proper splits"""
    
    def __init__(self, dataset_name='MNIST', data_dir='./data', batch_size=32, 
                 val_split=0.1, test_split=0.1, seed=42):
        """
        Args:
            dataset_name: 'MNIST', 'FashionMNIST', or 'CIFAR10'
            data_dir: Directory to store datasets
            batch_size: Batch size for training
            val_split: Validation set proportion (from training data)
            test_split: Test set proportion (from training data)
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self._load_dataset()
    
    def _get_transforms(self):
        """Get appropriate transforms for each dataset"""
        if self.dataset_name in ['MNIST', 'FashionMNIST']:
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        elif self.dataset_name == 'CIFAR10':
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                   (0.2023, 0.1994, 0.2010))
            ])
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        return train_transform, test_transform
    
    def _load_dataset(self):
        """Load and split dataset with retry logic for downloads"""
        train_transform, test_transform = self._get_transforms()
        
        # Load dataset
        dataset_class = getattr(datasets, self.dataset_name)
        
        # Load full training set with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                full_train = dataset_class(
                    root=self.data_dir,
                    train=True,
                    transform=train_transform,
                    download=True
                )
                break
            except (ConnectionError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Download attempt {attempt + 1} failed: {e}")
                    print(f"   Retrying... ({attempt + 2}/{max_retries})")
                    import time
                    time.sleep(2)
                else:
                    print(f"❌ Failed to download after {max_retries} attempts")
                    print("💡 Tip: You can manually download MNIST from:")
                    print("   http://yann.lecun.com/exdb/mnist/")
                    raise
        
        # Load test set with retry
        for attempt in range(max_retries):
            try:
                original_test = dataset_class(
                    root=self.data_dir,
                    train=False,
                    transform=test_transform,
                    download=True
                )
                break
            except (ConnectionError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Download attempt {attempt + 1} failed: {e}")
                    print(f"   Retrying... ({attempt + 2}/{max_retries})")
                    import time
                    time.sleep(2)
                else:
                    print(f"❌ Failed to download after {max_retries} attempts")
                    raise
        
        # Calculate split sizes
        total_size = len(full_train)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size - test_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_train, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        # Store datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.original_test = original_test
        
        # Dataset info
        self.num_classes = len(full_train.classes)
        if self.dataset_name in ['MNIST', 'FashionMNIST']:
            self.input_shape = (1, 28, 28)
        else:  # CIFAR10
            self.input_shape = (3, 32, 32)
        
        print(f"✓ Loaded {self.dataset_name}")
        print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
        print(f"  Classes: {self.num_classes} | Input shape: {self.input_shape}")
    
    def get_loaders(self, batch_size=None, num_workers=4):
        """Get data loaders for train, validation, and test sets"""
        if batch_size is None:
            batch_size = self.batch_size
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_info(self):
        """Return dataset information"""
        return {
            'name': self.dataset_name,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train_size': len(self.train_dataset),
            'val_size': len(self.val_dataset),
            'test_size': len(self.test_dataset)
        }


# Quick test function
if __name__ == "__main__":
    # Test loader
    loader = DatasetLoader('MNIST', batch_size=64)
    train_loader, val_loader, test_loader = loader.get_loaders()
    
    # Print batch info
    x, y = next(iter(train_loader))
    print(f"\nBatch shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Data type: {x.dtype}")