"""
Modern CNN architecture with batch normalization and dropout
"""
import torch
import torch.nn as nn


class ModernCNN(nn.Module):
    """
    Convolutional Neural Network with modern techniques
    """
    
    def __init__(self, input_shape=(1, 28, 28), num_classes=10,
                 conv_channels=[32, 64], fc_hidden=128, dropout_rate=0.3):
        """
        Args:
            input_shape: (C, H, W) input dimensions
            num_classes: Number of output classes
            conv_channels: List of convolutional layer output channels
            fc_hidden: Hidden size for fully connected layer
            dropout_rate: Dropout probability
        """
        super(ModernCNN, self).__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Convolutional layers
        in_channels = input_shape[0]
        conv_layers = []
        
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Calculate flattened size
        self.flat_size = conv_channels[-1] * 4 * 4
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_size, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_from_config(input_shape, num_classes, config):
    """
    Create CNN from hyperparameter configuration
    
    Args:
        input_shape: Input dimensions
        num_classes: Number of classes
        config: Dict with 'conv_channels', 'fc_hidden', 'dropout'
    """
    conv_channels = config.get('conv_channels', [32, 64])
    fc_hidden = config.get('fc_hidden', 128)
    dropout = config.get('dropout', 0.3)
    
    model = ModernCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        conv_channels=conv_channels,
        fc_hidden=fc_hidden,
        dropout_rate=dropout
    )
    
    return model


# Test the model
if __name__ == "__main__":
    # Test for MNIST
    model = ModernCNN(input_shape=(1, 28, 28), num_classes=10)
    print(f"Model: {model}")
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(32, 1, 28, 28)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")