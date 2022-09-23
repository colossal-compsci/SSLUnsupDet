import torch
from torch.utils.data import Dataset

import numpy as np
import soundfile as sf
import librosa
import random

from preprocess import *

class SpermWhaleClicks(Dataset):
    def __init__(self,
                 n_samples,
                 subset,
                 base_path='data/wavs/*.wav',
                 window=0.5,
                 window_pad=136,
                 sample_rate=48000,
                 seed=42,
                 epsilon=2e-6):
        self.n_samples = n_samples
        self.subset = subset
        self.window = window
        self.window_pad = window_pad
        self.sample_rate = sample_rate
        self.seed = seed
        self.epsilon = epsilon
        
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        self.preprocesser = WatkinsSpermWhalePreprocess(base_path=base_path,
                                                        window_width=self.window + self.window_pad / self.sample_rate)
        self.wavs, self.weights = self.preprocesser.run()
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self.subset != 'train':
            np.random.seed(idx)
            random.seed(idx)
            
        wav = np.random.choice(self.wavs, p=self.weights)
        dur = sf.info(wav).duration
        
        start_time = random.uniform(0, dur - (self.window + self.window_pad / self.sample_rate + self.epsilon))
        
        x, _ = librosa.load(wav, 
                            offset=start_time,
                            duration=self.window + self.window_pad / self.sample_rate+self.epsilon,
                            sr=self.sample_rate)
        window_frames = int(self.window * self.sample_rate) + self.window_pad
        assert x.shape[0] >= window_frames
        x = librosa.util.fix_length(x, window_frames)
        x = torch.tensor(x).unsqueeze(dim=0)
        return x, x