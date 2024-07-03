# classification_models.py
import torch

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=1, pretrained=True, weights_path=None):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet50(pretrained=False)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)
        return x

