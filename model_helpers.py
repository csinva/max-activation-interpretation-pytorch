import torch

class Model_feats(torch.nn.Module):
    def __init__(self, model, model_name, feat_size=4096, device='cuda'):
        super(Model_feats, self).__init__()        
        self.feats = torch.zeros(1, feat_size, requires_grad=True, device=device)
        self.model = model
        self.register_forward(model, model_name)
    
    # extract features from one of the last layers    
    def register_forward(self, model, model_name):
        if model_name == 'alexnet' or 'vgg' in model_name:
            layer = model.features[-1]
        elif 'resnet' in model_name:
            layer = model.fc
        elif 'densenet' in model_name:
            layer = model.classifier

        # features will be the input of the selected layer
        def copy_data(m, i, o):
            self.feats = i[0] # i is a tuple
        layer.register_forward_hook(copy_data)
        
    def forward(self, x):
        self.model(x)
        return self.feats