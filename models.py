import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers import *

class SpectralBoundaryEncoder(nn.Module):
    def __init__(self, 
                 preprocess_params,
                 transform_params,
                 postprocess_params):
        super(SpectralBoundaryEncoder, self).__init__()
        
        self.preprocess_params = preprocess_params
        self.transform_params = transform_params
        self.postprocess_params = postprocess_params
        
        try:
            filter_params = self.preprocess_params['filter_params']
            cutoff_freq = filter_params['cutoff_freq']
            b = filter_params['b']
            sr = filter_params['sample_rate']
            ramp_duration = filter_params['ramp_duration']
            self.preprocess = HighPassFilter(cutoff_freq=cutoff_freq,
                                             sample_rate=sr,
                                             b=b,
                                             ramp_duration=ramp_duration)
        except KeyError:
            self.preprocess = IdentityLayer()

        if self.transform_params['name'] == 'stft':
            params = self.transform_params['params']
            self.transform = STFT(**params)
            if self.postprocess_params['lambd'] == 'concat':
                lambd = lambda x: torch.cat(x, dim=1).transpose(1,2)
            elif self.postprocess_params['lambd'] == 0:
                lambd = lambda x: x[0].transpose(1,2)
            elif self.postprocess_params['lambd'] == 1:
                lambd = lambda x: x[1].transpose(1,2)
            self.postprocess = LambdaLayer(lambd=lambd)
        elif 'conv1d' in self.transform_params['name']:
            params = self.transform_params['params']
            self.transform = ConvTransform(**params)
            mlp_params = self.postprocess_params['mlp_params']
            lambd = lambda x: x.transpose(1, 2)
            in_dimension = self.transform.latent_dim
            
            output_activation = self.postprocess_params['output_activation']
            if output_activation is None:
                out_activation = IdentityLayer()
            elif output_activation == 'tanh':
                out_activation = nn.Tanh()
            elif output_activation == 'sigmoid':
                out_activation = nn.Sigmoid()
            elif output_activation == 'hardtanh':
                out_activation = nn.HardTanh()
                
            if mlp_params is not None:
                self.postprocess = nn.Sequential(LambdaLayer(lambd=lambd),
                                                 MLP(in_dimension=in_dimension,
                                                     **mlp_params),
                                                 out_activation)
            else:
                self.postprocess = nn.Sequential(LambdaLayer(lambd=lambd),
                                                 out_activation)
                
    def forward(self, x):
        x = self.preprocess(x)
        x = self.transform(x)
        z = self.postprocess(x)
        return z
    
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(SpectralBoundaryEncoder, self).__str__() + f'\nTrainable parameters: {params}'