import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import defaultdict

from utils import *

class NoiseContrastiveEstimationMetric(object):
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
                time_reorder = torch.arange(positive_pred.shape[1])
                
                negative_pred = self.similarity(z[:, :-t], z[: , time_reorder])
                preds[t].append(negative_pred)
        return preds
    
    def metric(self, preds):
        m = 0
        for t, t_preds in preds.items():
            out = torch.stack(t_preds, dim=-1)
            out = F.log_softmax(out, dim=-1)
            out = out[..., 0]
            m += -out.mean()
        return m

class PRF1Metric(object):
    def __init__(self,
                 tolerance):
        self.tolerance = tolerance
        
    def recall(self, true_positives, false_negatives):
        return true_positives / (true_positives + false_negatives)
    
    def precision(self, true_positives, false_positives):
        return true_positives / (true_positives + false_positives)
    
    def f1(self, true_positives, false_negatives, false_positives):
        return 2 * true_positives / (2 * true_positives + false_positives + false_negatives)
    
    def prob_detection(self, true_positives, trues):
        return true_positives / len(trues)
    
    def prob_false_alarm(self, true_positives, false_positives):
        return false_positives / (false_positives + true_positives)
    
    def classify_predictions(self, trues, preds):
        true_positives = 0
        false_negatives = 0
        for t in trues:
            condition = np.logical_and(t - self.tolerance <= preds, preds <= t + self.tolerance)
            if len(preds[condition]) > 0:
                true_positives += 1
            else:
                false_negatives += 1
                
        false_positives = 0
        for p in preds:
            condition = np.logical_and(p - self.tolerance <= trues, trues <= p + self.tolerance)
            if len(trues[condition]) > 0:
                pass
            else:
                false_positives += 1
                
        return true_positives, false_negatives, false_positives
    
    def r_value(self, R, P):
        os = R / P - 1
        r1 = np.sqrt((1 - R) ** 2 + os ** 2)
        r2 = (-os + R - 1) / (np.sqrt(2))
        r_val = 1 - (np.abs(r1) + np.abs(r2)) / 2
        return r_val
    
    def compute_metrics(self, trues, preds):
        TP, FN, FP = self.classify_predictions(trues, preds)
        
        R = self.recall(TP, FN)
        P = self.precision(TP, FP)
        F1 = self.f1(TP, FN, FP)
        P_det = self.prob_detection(TP, trues)
        P_FA = self.prob_false_alarm(TP, FP)
        R_Val = self.r_value(R, P)
        
        metrics = {'Recall':R,
                   'Precision':P,
                   'F1':F1,
                   'R-Value':R_Val,
                   'P_detection':P_det,
                   'P_false_alarm':P_FA}
        return metrics