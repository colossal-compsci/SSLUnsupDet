import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import glob

class WatkinsSpermWhalePreprocess(object):
    def __init__(self, 
                 base_path='data/wavs/*.wav',
                 window_width=0.5):
        self.wav_files = sorted(glob.glob(base_path))
        self.window_width = window_width
        
    def get_info(self):
        srs = []
        durs = []
        wavs = []
        for w in self.wav_files:
            info = sf.info(w)
            sr = info.samplerate
            dur = info.duration
            if dur > self.window_width:
                srs.append(info.samplerate)
                durs.append(info.duration)
                wavs.append(w)
        return wavs, durs, srs
    
    def run(self):
        w, d, _ = self.get_info()
        weights = self.probability_weights(d)
        return w, weights
        
    @staticmethod
    def probability_weights(x):
        if type(x) != np.ndarray:
            x = np.array(x)
        weights = x / sum(x)
        return weights