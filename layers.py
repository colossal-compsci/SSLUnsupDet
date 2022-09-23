import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import numpy as np
import tqdm

class HighPassFilter(nn.Module):
    def __init__(self, cutoff_freq, sample_rate, b=0.08, eps=1e-20, ramp_duration=None):
        super(HighPassFilter, self).__init__()
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.fc = cutoff_freq / sample_rate
        self.b = b

        N = int(np.ceil((4 / b)))
        if not N % 2:
            N+=1
        self.N = N

        self.epsilon = nn.Parameter(torch.tensor(eps), requires_grad=False)
        self.window = nn.Parameter(torch.blackman_window(N), requires_grad=False)
        
        n = torch.arange(N)
        self.sinc_fx = nn.Parameter(self.sinc(2 * self.fc * (n - (self.N-1) / 2.)), requires_grad=False)

        self.ramp_duration = ramp_duration
        if self.ramp_duration is not None:
            self.ramp = nn.Parameter(self.hann_ramp(sample_rate, ramp_duration), requires_grad=False)
        else:
            self.ramp = None

    def forward(self, x):
        size = x.size()
        x = x.view(x.size(0), 1, x.size(-1))
        sinc_fx = self.sinc_fx * self.window
        sinc_fx = torch.true_divide(sinc_fx, torch.sum(sinc_fx))
        sinc_fx = -sinc_fx
        sinc_fx[int((self.N - 1) / 2)] += 1
        output = torch.nn.functional.conv1d(x, sinc_fx.view(-1, 1, self.N), padding=self.N//2)
        if self.ramp is not None:
            output[:, :, :len(self.ramp)] = output[:, :, :len(self.ramp)] * torch.flip(self.ramp, [0])
            output[:, :, -len(self.ramp):] = output[:, :, -len(self.ramp):] * self.ramp
        return output.reshape(size)

    def sinc(self, x):
        y = np.pi*torch.where(x==0, self.epsilon, x)
        return torch.true_divide(torch.sin(y), y)  

    def get_config(self):
        config  = {
            'name': 'HighPassFilter',
            'cutoff_freq': self.cutoff_freq,
            'sample_rate': self.sample_rate,
            'b':self.b
        }
        return config

    @staticmethod
    def hann_ramp(sample_rate, ramp_duration=0.002):
        t = np.arange(start=0, stop=ramp_duration, step=1/sample_rate)
        off_ramp = 0.5*(1. + np.cos( (np.pi/ramp_duration)*t )).astype('float32')
        return torch.tensor(off_ramp)

class STFT(nn.Module):
    def __init__(self, 
                 kernel_size, 
                 stride, 
                 coords='polar',
                 dB=False,
                 center=True,
                 epsilon=1e-8):
        super(STFT, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.register_buffer("window", torch.hann_window(self.kernel_size))
        self.coords = coords
        self.epsilon = epsilon
        self.dB = dB
        self.center = center

        if self.dB:
            assert self.coords=='polar', 'dB requires magnitude spectrogram'

    def forward(self, x):
        S = torch.stft(x.squeeze(dim=1), 
                       n_fft=self.kernel_size, 
                       hop_length=self.stride, 
                       window=self.window,
                       onesided=True,
                       center=self.center,
                       pad_mode='reflect',
                       normalized=False,
                       return_complex=False)
        S_real = S[:, :, :, 0]
        S_imag = S[:, :, :, 1]
        if self.coords == 'cartesian':
            return S_real, S_imag
        elif self.coords == 'polar':
            S_real = S_real + self.epsilon
            S_imag = S_imag + self.epsilon
            S_phase = torch.atan2(S_imag, S_real)
            S_mag = torch.sqrt(torch.add(torch.pow(S_real, 2), torch.pow(S_imag, 2)))
            if self.dB:
                S_mag = self.amplitude_to_db(S_mag)
            return S_phase, S_mag

    def get_out_size(self, in_size):
        batch, in_filters, L_in = in_size
        if self.center:
            L_out = L_in // self.stride + 1
        else:
            L_out = (L_in - self.kernel_size) // self.stride + 1
        out_filters = self.kernel_size // 2 + 1
        return (batch, out_filters, L_out)

    def get_config(self):
        config = {
            'name': 'STFT',
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dB scaling': self.dB
        }
        return config

    @staticmethod
    def amplitude_to_db(S, amin=1e-5, delta_db=80):
        S[S < amin] = amin
        D = torch.mul(torch.log10(S), 20)
        if delta_db is not None:
            D[D < D.max() - delta_db] = D.max() - delta_db
        return D

class iSTFT(nn.Module):
    def __init__(self, 
                 kernel_size, 
                 stride, 
                 coords='polar',
                 dB=False,
                 center=True):
        super(iSTFT, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.register_buffer("window", torch.hann_window(self.kernel_size))
        self.coords = coords
        self.dB = dB

        if self.dB:
            assert self.coords=='polar', 'dB requires magnitude spectrogram'

    def forward(self, S1, S2):
        if self.coords == 'cartesian':
            S_real, S_imag = S1.unsqueeze(dim=-1), S2.unsqueeze(dim=-1)
        elif self.coords == 'polar':
            S_phase, S_mag = S1, S2
            if self.dB:
                S_mag = self.db_to_amplitude(S_mag)
            S_real = torch.mul(S_mag, torch.cos(S_phase)).unsqueeze(dim=-1)
            S_imag = torch.mul(S_mag, torch.sin(S_phase)).unsqueeze(dim=-1)
        S = torch.cat([S_real, S_imag], dim=-1)

        x = torch.istft(S, 
                        n_fft=self.kernel_size, 
                        hop_length=self.stride, 
                        window=self.window,
                        return_complex=False).unsqueeze(dim=1)
        return x

    def get_out_size(self, in_size):
        batch, in_filters, L_in = in_size
        L_out = int(L_in - 1) * self.stride
        if not self.center:
            L_out += self.kernel_size // 2
        return (batch, 1, L_out)

    def get_config(self):
        config = {
            'name': 'iSTFT',
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dB scaling': self.dB
        }
        return config

    @staticmethod
    def db_to_amplitude(D, amin=1e-10):
        S = torch.pow(10, torch.true_divide(D, 20)) - amin
        return S

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()
    def forward(self, x):
        return x

class MLP(nn.Module):
    def __init__(self,
                 in_dimension,
                 out_dimension,
                 dropout,
                 linear_projection=True):
        super(MLP, self).__init__()
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        if self.out_dimension is None:
            self.out_dimension = self.in_dimension
        self.dropout = dropout
        self.linear_projection = linear_projection

        if self.linear_projection:
            layers = [nn.Dropout2d(self.dropout),
                      nn.Linear(self.in_dimension, self.out_dimension)]
        else:
            layers = [nn.Dropout2d(self.dropout),
                      nn.Linear(self.in_dimension, self.in_dimension),
                      nn.LeakyReLU(),
                      nn.Dropout2d(self.dropout),
                      nn.Linear(self.in_dimension, self.out_dimension)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(MLP, self).__str__() + f'\nTrainable parameters: {params}'

class ConvTransform(nn.Module):
    def __init__(self,
                 kernel_size,
                 stride,
                 encoder_name='conv1d',
                 out_channels=None,
                 center=False,
                 double_channels=False,
                 layers=None,
                 bias=False,
                 affine=False):
        super(ConvTransform, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.encoder_name = encoder_name
        self.out_channels = out_channels
        self.center = center
        self.double_channels = double_channels
        self.layers = layers
        self.bias = bias
        self.affine = affine
        
        if self.center:
            self.padding_params = {
                'padding': self.kernel_size // 2,
                'padding_mode': 'reflect'
            }
        else:
            self.padding_params = {
                'padding': 0
            }
        
        if self.out_channels is None:
            self.out_channels = self.kernel_size // 2 + 1
        
        if self.double_channels:
            self.out_channels *= 2
        
        if self.encoder_name == 'conv1d':
            self.transform = nn.Sequential(nn.Conv1d(in_channels=1,
                                                     out_channels=self.out_channels,
                                                     kernel_size=self.kernel_size,
                                                     stride=self.stride,
                                                     bias=self.bias,
                                                     **self.padding_params),
                                           nn.BatchNorm1d(self.out_channels,
                                                          affine=self.bias),
                                           nn.LeakyReLU())
        
        elif self.encoder_name == 'stft_db':
            self.transform = nn.Sequential(STFT(kernel_size=self.kernel_size,
                                                stride=self.stride,
                                                dB=True,
                                                coords='polar',
                                                center=self.center),
                                           LambdaLayer(lambd=lambda x: x[1]))
        elif self.encoder_name == 'conv1d_layers':
            #assert self.kernel_size in [92, 465, 905], 'Choose a different kernel size'
            #assert self.stride in [32, 160, 320], 'Choose a different stride'
            
            #if self.stride == 32:
                #conv1d_layers = [[8, 4, 0], [4, 2, 0], [4, 2, 0], [4, 2, 0]]
            #elif self.stride == 160:
                #conv1d_layers = [[10, 5, 0], [8, 4, 0], [4, 2, 0], [4, 2, 0], [4, 2, 0]]
            #elif self.stride == 320:
                #conv1d_layers = [[10, 5, 0], [8, 4, 0], [8, 4, 0], [4, 2, 0], [4, 2, 0]]
            #conv1d_layers = [[8, 4, 0], [6, 3, 0], [4, 2, 0], [4, 2, 0], [4, 2, 0]]
            conv1d_layers = [[8, 4, 0], [6, 3, 0], [4, 2, 0], [4, 2, 0]]
                
            transform_layers = [nn.Conv1d(in_channels=1,
                                          out_channels=self.out_channels,
                                          kernel_size=conv1d_layers[0][0],
                                          stride=conv1d_layers[0][1],
                                          padding=conv1d_layers[0][2],
                                          bias=self.bias),
                               nn.BatchNorm1d(self.out_channels,
                                              affine=self.affine),
                               nn.LeakyReLU()]
            for l in range(1, len(conv1d_layers)):
                transform_layers.append(nn.Conv1d(in_channels=self.out_channels,
                                                  out_channels=self.out_channels,
                                                  kernel_size=conv1d_layers[l][0],
                                                  stride=conv1d_layers[l][1],
                                                  padding=conv1d_layers[l][2],
                                                  bias=self.bias))
                transform_layers.append(nn.BatchNorm1d(self.out_channels,
                                                       affine=self.affine))
                transform_layers.append(nn.LeakyReLU())

            self.transform = nn.Sequential(*transform_layers)
            
        self.latent_dim = self.out_channels
        
        if self.layers is not None:
            in_channel = self.out_channels
            out_channel = self.out_channels
            for i, layer in enumerate(self.layers):
                if len(layer) == 4:
                    in_channel = out_channel
                    out_channel, kernel, stride, padding = layer
                elif len(layer) == 3:
                    kernel, stride, padding = layer
                    
                assert stride==1, 'Choose stride=1'
                assert padding == int((kernel - 1) / 2), 'Choose a new padding'
                self.transform.add_module(f'conv{i+1}', nn.Conv1d(in_channels=in_channel,
                                                                  out_channels=out_channel,
                                                                  kernel_size=kernel,
                                                                  stride=stride,
                                                                  padding=padding,
                                                                  bias=self.bias))
                self.transform.add_module(f'bn{i+1}', nn.BatchNorm1d(out_channel,
                                                                     affine=self.affine))
                self.transform.add_module(f'act{i+1}', nn.LeakyReLU())
                in_channel = out_channel
            self.latent_dim = out_channel
            
    def forward(self, x):
        return self.transform(x)

    def get_out_size(self, in_size):
        batch, in_filters, L_in = in_size
        if self.encoder_name in ['conv1d', 'stft']:
            if self.center:
                L_out = L_in // self.stride + 1
            else:
                L_out = (L_in - self.kernel_size) // self.stride + 1
            out_filters = self.kernel_size // 2 + 1
        else:
            out = self.forward(torch.ones(1, in_filters, L_in))
            out_filters = out.size(1)
            L_out = out.size(-1)
        return (batch, out_filters, L_out)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(ConvTransform, self).__str__() + f'\nTrainable parameters: {params}'