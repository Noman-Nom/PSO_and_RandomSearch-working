"""
Professional training utilities with early stopping and validation
"""
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from tqdm import tqdm
import numpy as np
import time


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001, verbose=False):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
    
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class ModelTrainer:
    """
    Professional model trainer with validation and early stopping
    """
    
    def __init__(self, model, device='cuda', learning_rate=0.001, 
                 optimizer_type='adam', weight_decay=1e-4):
        """
        Args:
            model: PyTorch model
            device: Device to train on
            learning_rate: Initial learning rate
            optimizer_type: 'adam' or 'sgd'
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = Adam(model.parameters(), lr=learning_rate, 
                                weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = SGD(model.parameters(), lr=learning_rate, 
                               momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, 
              early_stopping_patience=10, verbose=True):
        """
        Train model with validation and early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Print progress
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience, 
                                      verbose=False)
        
        best_val_acc = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch+1}/{epochs}: '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        training_time = time.time() - start_time
        
        return {
            'best_val_acc': best_val_acc,
            'final_train_acc': train_acc,
            'final_val_acc': val_acc,
            'epochs_trained': epoch + 1,
            'training_time': training_time,
            'history': self.history
        }
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        return accuracy


def train_model_with_config(model, train_loader, val_loader, config, device='cuda'):
    """
    Convenience function to train model with hyperparameter config
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Dict with 'learning_rate', 'optimizer', 'epochs', etc.
        device: Training device
    
    Returns:
        dict: Training results
    """
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=config.get('learning_rate', 0.001),
        optimizer_type=config.get('optimizer', 'adam'),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get('epochs', 50),
        early_stopping_patience=config.get('patience', 10),
        verbose=config.get('verbose', False)
    )
    
    return results, trainer


# Quick test
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    # Dummy data
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(train_data, batch_size=64, shuffle=False)
    
    # Dummy model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Train
    trainer = ModelTrainer(model, device='cpu', learning_rate=0.001)
    results = trainer.train(train_loader, val_loader, epochs=2, verbose=True)
    print(f"\nResults: {results}")