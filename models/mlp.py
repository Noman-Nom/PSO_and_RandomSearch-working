"""
Modern MLP with BatchNorm, Dropout, and proper initialization
"""
import torch
import torch.nn as nn


class ModernMLP(nn.Module):
    """
    Multi-Layer Perceptron with modern deep learning techniques
    """
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10, 
                 hidden_sizes=[256, 128], dropout_rate=0.3, use_batchnorm=True):
        """
        Args:
            input_shape: (C, H, W) input dimensions
            num_classes: Number of output classes
            hidden_sizes: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batchnorm: Whether to use batch normalization
        """
        super(ModernMLP, self).__init__()
        
        self.input_shape = input_shape
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_size = self.input_dim
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(nn.ReLU(inplace=True))
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He initialization for ReLU networks"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.network(x)
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_mlp_from_config(input_shape, num_classes, config):
    """
    Create MLP from hyperparameter configuration
    
    Args:
        input_shape: Input dimensions
        num_classes: Number of classes
        config: Dict with 'hidden_size', 'num_layers', 'dropout', 'use_batchnorm'
    """
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 2)
    dropout = config.get('dropout', 0.3)
    use_batchnorm = config.get('use_batchnorm', True)
    
    # Create hidden sizes list
    hidden_sizes = [hidden_size] * num_layers
    
    model = ModernMLP(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout,
        use_batchnorm=use_batchnorm
    )
    
    return model


# Test the model
if __name__ == "__main__":
    # Test for MNIST
    model = ModernMLP(input_shape=(1, 28, 28), num_classes=10, hidden_sizes=[256, 128])
    print(f"Model: {model}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(32, 1, 28, 28)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")