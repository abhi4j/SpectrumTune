import torch
import torch.nn as nn
import torchvision.models as models

class GenreClassifier(nn.Module):
    """ResNet50-based model for music genre classification with configurable layer freezing"""
    
    def __init__(self, num_classes=10, pretrained=True, freeze_strategy='all'):
        """
        Args:
            num_classes (int): Number of output classes (genres)
            pretrained (bool): Whether to use pretrained weights
            freeze_strategy (str): One of 'all', 'layer4', 'layer3', 'layer2', 'none'
        """
        super(GenreClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Apply freezing based on strategy
        self.apply_freezing_strategy(freeze_strategy)
        
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def apply_freezing_strategy(self, strategy):
        """Apply the specified freezing strategy to the model"""
        # First unfreeze everything
        for param in self.model.parameters():
            param.requires_grad = True
            
        if strategy == 'all':
            # Freeze all layers except final FC
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
        
        elif strategy == 'layer4':
            # Freeze all layers except layer4 and FC
            for name, param in self.model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
        
        elif strategy == 'layer3':
            # Freeze all layers except layer3, layer4 and FC
            for name, param in self.model.named_parameters():
                if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
        
        elif strategy == 'layer2':
            # Freeze all layers except layer2, layer3, layer4 and FC
            for name, param in self.model.named_parameters():
                if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False
        
        elif strategy == 'none':
            # Don't freeze any layers - already set all to trainable above
            pass
        
        else:
            raise ValueError(f"Unsupported freeze strategy: {strategy}")
    
    def print_trainable_parameters(self):
        """Print which parameters are trainable and the total count"""
        trainable_params = 0
        all_params = 0
        
        # Print each trainable parameter
        for name, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable: {name}")
        
        print(f"\nTrainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        print(f"Frozen parameters: {all_params - trainable_params:,} ({100 * (all_params - trainable_params) / all_params:.2f}%)")
        print(f"Total parameters: {all_params:,}")