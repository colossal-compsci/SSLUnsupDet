import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from utils import *

class NoiseContrastiveEstimationLoss(object):
    def __init__(self,
                 pred_steps=1,
                 pred_offset=0,
                 n_negatives=1,
                 sim_metric_params={
                     'name': 'cosine'
                 }):
        
        self.pred_steps = list(range(1 + pred_offset, 1 + pred_offset + pred_steps))
        self.pred_offset = pred_offset
        self.n_negatives = n_negatives
        
        if sim_metric_params['name'] == 'cosine':
            self.similarity_metric = lambda z1, z2: F.cosine_similarity(z1, z2, dim=-1)
        elif sim_metric_params['name'] == 'bounded_euclidean':
            self.similarity_metric = lambda z1, z2: bounded_euclidean_similarity(z1, z2,
                                                                                 dim=-1,
                                                                                 **sim_metric_params['params'])
        
    def similarity(self, z, z_shift):
        return self.similarity_metric(z, z_shift)
        
    def compute_preds(self, z):
        preds = defaultdict(list)
        for i, t in enumerate(self.pred_steps):
            positive_pred = self.similarity(z[:, :-t], z[:, t:])
            preds[t].append(positive_pred)
            
            for _ in range(self.n_negatives):
                time_reorder = torch.randperm(positive_pred.shape[1])
                
                negative_pred = self.similarity(z[:, :-t], z[: , time_reorder])
                preds[t].append(negative_pred)
        return preds
    
    def loss(self, preds):
        l = 0
        for t, t_preds in preds.items():
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            out = out[..., 0]
            l += -out.mean()
        return l