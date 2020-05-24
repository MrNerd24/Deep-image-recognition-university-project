

import torch


class EnsembleModel(torch.nn.Module):

    def __init__(self, device, models, modelsOutputNeurosNum, outputNeurosNum):
        super(EnsembleModel, self).__init__()
        
        self.models = torch.nn.ModuleList(models)
        liearInputLength = sum(modelsOutputNeurosNum)
        self.linear = torch.nn.Linear(liearInputLength, outputNeurosNum)
        self.sigmoid = torch.nn.Sigmoid()
        self.to(device)

    def forward(self, x):
        modelOutputs = []
        for model in self.models:
            modelOutputs.append(model(x))
        
        output = torch.cat(modelOutputs, 1)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output



class IdentityLayer(torch.nn.Module):

    def forward(self, x):
        return x

