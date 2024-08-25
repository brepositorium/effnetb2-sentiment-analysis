import torch
import torchvision

def create_effnetb2_model(num_classes: int) -> torch.nn.Module:
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)
    
    for param in model.features.parameters():
        param.requires_grad = False
    
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features=1408, out_features=num_classes, bias=True)
    )
    
    return model

def get_transforms():
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    return weights.transforms()