from args import get_args
import torch
import torch.nn as nn 
import torchvision.models as models 


class MyModel(torch.nn.Module):
    def __init__(self, backbone = 'resnet18', num_classes = 10):
        super(MyModel,self).__init__()

        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained = True, num_classes=5)
        
        elif backbone == 'resnet34':
            self.model = models.resnet34 (num_classes=5)

        else:
            self.model = models.resnet50(num_classes=5)   

        
        #self.model.fc =

    def forward(self,x):
        return self.model(x)
