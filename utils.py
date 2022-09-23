import torch
import torch.nn.functional as F
import numpy as np
import librosa
import random
import tqdm
import re

"""
def collate_pad(batch, padding=-1):
    data = [b[0] for b in batch]
    target = [b[1] for b in batch]
    target = torch.nn.utils.rnn.pad_sequence(target,
                                             batch_first=True,
                                             padding_value=padding)
    mask = (target != -1)
    return torch.stack(data), target, mask
"""

def bounded_euclidean_similarity(x, y, alpha=1, beta=0, dim=-1):
    delta = torch.pow(x - y, 2)
    d = torch.sum(delta, dim=dim)
    return 1 - torch.tanh(F.relu(alpha*d + beta))

def spectral_distances(z, metric='cosine', front_pad=False, **kwargs):
    z1 = z[:, :-1]
    z2 = z[:, 1:]
    if metric=='cosine':
        d = 1 - F.cosine_similarity(z1, z2, dim=-1)
    elif metric=='bounded_euclidean':
        d = 1 - bounded_euclidean_similarity(z1, 
                                             z2, 
                                             alpha=kwargs['alpha'], 
                                             beta=kwargs['beta'], 
                                             dim=-1)
    d = d.squeeze().numpy()
    if front_pad:
        d = np.insert(d, 0, d[0])
    return d

def box_convolve(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

"""
def overlap_average(x, y, window, overlap=None):
    if overlap is None:
        if window % 2 == 0:
            overlap = window // 2
        else:
            overlap = window // 2 + 1
    
    assert y.shape[0] == window
    y = np.pad(y, 
               (x.shape[0] - overlap, 0), 
               mode='constant', 
               constant_values=0)
    x = np.pad(x, 
               (0, window - overlap), 
               mode='constant', 
               constant_values=0)
    
    x[-window:-window+overlap] *= 1 / 2 
    y[-window:-window+overlap] *= 1 / 2 
    return x + y

def box_convolve(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def run_inference(dataloader, 
                  model, 
                  peak_detector, 
                  metrics,
                  return_audios=False,
                  device='cpu'):
    TP, FP, FN = 0, 0, 0
    D, N_clicks = 0, 0
    all_audios, all_peaks = [], []
    
    N_codas = len(dataloader)
    for i, data in tqdm.tqdm(enumerate(dataloader), total=N_codas):
        audio = data[0].to(device)
        times = data[-1]
        times = np.array([t.item() for t in times])
        
        peaks = peak_detector.detect_peaks(audio, model)
        tp, fn, fp = metrics.classify_predictions(times, peaks)
        
        TP += tp
        FN += fn
        FP += fp
        N_clicks += len(times)
        D += peak_detector.sample_to_time(audio.size(-1) - peak_detector.window_padding)
        all_audios.append(audio)
        all_peaks.append(peaks)
        
    recall = metrics.recall(TP, FN)
    precision = metrics.precision(TP, FP)
    f1 = metrics.f1(TP, FN, FP)
    r_value = metrics.r_value(recall, precision)
    p_det = metrics.prob_detection(TP, [_ for _ in range(N_clicks)])
    p_fa = metrics.prob_false_alarm(TP, FP)
    
    results = {}
    results['Total Clicks'] = N_clicks
    results['Total Codas'] = N_codas
    results['Total Duration (s)'] = D
    results['Recall'] = recall
    results['Precision'] = precision
    results['F1'] = f1
    results['R-Value'] = r_value
    results['P_Detection'] = p_det
    results['P_False_Alarm'] = p_fa
    if return_audios:
        return results, audios, peaks
    else:
        return results
"""    
def range_step(in_list):
    if len(in_list) == 3:
        return np.arange(*in_list)
    else:
        #assert in_list[0] is None
        #return [None]
        assert len(in_list) == 1
        return in_list
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname != 'ConvTransform':
        torch.nn.init.xavier_uniform_(m.weight)

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_filename(wav):
    return re.split(r'\\', wav)[-1][:-4]
    
class ArchitectureBuilder(object):
    
    def __init__(self, 
                 net = [[8, 4, 0], [4, 2, 0], [4, 2, 0], [4, 2, 0]],
                 names = ['conv1', 'conv2', 'conv3', 'conv4'],
                 in_size=64000):
        self.net = net
        self.names = names
        self.in_size = in_size
    
    def build(self):
        current_layer = [self.in_size, 1, 1, 0.5]
        self.print_layer(current_layer, 'input')
        layer_infos = []
        for i in range(len(self.net)):
            current_layer = self.out_from_input(self.net[i], current_layer)
            layer_infos.append(current_layer)
            self.print_layer(current_layer, self.names[i])
            
    @staticmethod
    def print_layer(layer, layer_name):
        print(layer_name + ":")
        print(f'\t n_features: {layer[0]} \n \t jump: {layer[1]} \n \t receptive_size: {layer[2]} \t start: {layer[3]}')
    
    @staticmethod
    def out_from_input(conv, layer_in):
        n_in, j_in, r_in, start_in = layer_in
        k, s, p = conv

        n_out = math.floor((n_in - k + 2 * p) / s) + 1
        p_actual = (n_out - 1) * s - n_in + k
        p_R = math.ceil(p_actual / 2)
        p_L = math.floor(p_actual / 2)

        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - p_L) * j_in
        return n_out, j_out, r_out, start_out

def gen_spec(x, n, h, c=False):
    S = librosa.stft(x, n_fft=n, hop_length=h, center=c)
    D = librosa.amplitude_to_db(np.abs(S), ref=1)
    return D

def frame_to_time(frame, hop_length, sample_rate, shift=False):
    if shift:
        frame += 1
    return frame * (hop_length / sample_rate)

def sample_to_time(sample, sample_rate):
    return sample / sample_rate

def coarsen_selections(onsets, offsets, threshold):
    deltas = onsets[1:] - offsets[:-1]
    offset_indices = np.where(deltas < threshold)[0]
    onset_indices = offset_indices + 1
    onsets = np.delete(onsets, onset_indices)
    offsets = np.delete(offsets, offset_indices)
    return onsets, offsets

def add_lists(l1, l2):
    return [l1_i+l2_i for l1_i, l2_i in zip(l1, l2)]