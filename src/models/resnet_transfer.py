import torch
import torch.nn as nn
import torchvision.models as models

class GenreClassifier(nn.Module):
    """ResNet50-based model for music genre classification"""
    
    def __init__(self, num_classes=10, pretrained=True, freeze_layers=True):
        super(GenreClassifier, self).__init__()
        
        self.model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        
        if freeze_layers:

            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # Don't freeze the final fc layer
                    param.requires_grad = False
        
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
    # This method allows  to selectively unfreeze the 4th layer later if needed
    def unfreeze_layer4(self):
        """Unfreeze the fourth layer of ResNet for fine-tuning"""
        for name, param in self.model.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True
    
    # This method helps debug which parameters are actually being trained
    def print_trainable_parameters(self):
        """Print which parameters are trainable and the total count"""
        trainable_params = 0
        all_params = 0
        for name, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable: {name}")
        
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
        print(f"Total parameters: {all_params:,}")