import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FeatureExtractor(nn.Module):  
    def __init__(self, submodule, extracted_layers):  
        super(FeatureExtractor,self).__init__()  
        self.submodule = submodule  
        self.extracted_layers= extracted_layers  
  
    def forward(self, x):  
        outputs = list([])  
        for name, module in self.submodule._modules.items():  
            #if name is "fc": x = x.view(x.size(0), -1)  
            x = module(x)  
            #print(name)  
            if name in self.extracted_layers: 
                #print(x.size())
                x = x.view(x.size(0), -1).data.cpu().numpy()
                outputs.extend(x)
                return outputs
        return outputs  