import re 
import os
import glob
import json
import argparse
import pandas as pd

from metrics import *
from utils import *

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-c', '--config', type=str,
						help='JSON file for configuration')
	parser.add_argument('-q', '--use_q', type=int,
						help='use uncertain (q) selection tables (0 or 1)',
						default=0)
	parser.add_argument('-ckpt', '--checkpoint', type=int,
						help='choose the model ckpt')
	parser.add_argument('-s', '--search', type=str,
						help='choose the search regime (e.g. coarse or fine)')
	parser.add_argument('-e', '--compute_energy', type=int,
						help='compute energy detector baseline (0 or 1)',
						default=0)
	parser.add_argument('-hpf', '--use_hpf', type=int,
						help='use high pass filter (0 or 1)',
						default=0)

	args = parser.parse_args()
	"""
	class Args():
		def __init__(self, 
					 config='pipeline_v0.json',
					 use_q=0,
					 compute_energy=0):
			self.config = config
			self.use_q = use_q
			self.compute_energy = compute_energy

	args = Args()
	"""
	with open(f'configs/{args.config}') as f:
		data = f.read()
	f.close()
	config = json.loads(data)

	save_dir = config['utils']['save_dir']
	ckpt = args.checkpoint
	search = args.search
	assert search in ['coarse', 'fine']
	
	use_q = True if args.use_q == 1 else False

	compute_energy = True if args.compute_energy == 1 else False
	
	if compute_energy:
		preds_str = 'energy_'
		energy = 'energy_'
	else:
		preds_str = ''
		energy = ''

	use_hpf = args.use_hpf
	if use_hpf:
		preds_str += 'hpf_'
		hpf = 'hpf_'
	else:
		hpf = '' 

	if use_q:
		preds_str += f'predictions_ckpt{ckpt}_{search}*'
		q = '.q'
	else:
		preds_str += f'predictions_ckpt{ckpt}_{search}'
		q = ''

	prediction_tables = sorted(glob.glob(f'{save_dir}/Inference/*/{preds_str}.pkl'))

	for i, p in enumerate(prediction_tables):
		p = pd.read_pickle(p)
		
		current_onset_preds = p['OnsetPreds{coarsen:{tol:(TP,FN,FP)}}'].to_list()
		current_midpoint_preds = p['MidpointPreds{coarsen:{tol:(TP,FN,FP)}}'].to_list()
		
		current_n_signals = p['N_signals'].iloc[0]
		current_n_detections = p['N_detections'].to_list()

		if i == 0:
			coarsen_keys = list(current_onset_preds[0].keys())
			tolerance_keys = list(current_onset_preds[0][0].keys())
			n_grid_search = len(current_onset_preds)
			
			previous_onset_preds = current_onset_preds
			previous_midpoint_preds = current_midpoint_preds
			previous_n_signals = current_n_signals
			previous_n_detections = current_n_detections
		else:
			sum_onset_preds = []
			sum_midpoint_preds = []
			sum_n_signals = {}
			
			for j in tqdm.tqdm(range(n_grid_search)):
				on = {c:{} for c in coarsen_keys}
				mid = {c:{} for c in coarsen_keys}
				for c in coarsen_keys:
					sum_n_signals[c] = previous_n_signals[c] + current_n_signals[c]
					for t in tolerance_keys:
						on[c][t] = add_lists(previous_onset_preds[j][c][t], current_onset_preds[j][c][t])
						mid[c][t] = add_lists(previous_midpoint_preds[j][c][t], current_midpoint_preds[j][c][t])
				sum_onset_preds.append(on)
				sum_midpoint_preds.append(mid)
			sum_n_detections = add_lists(current_n_detections, previous_n_detections)
				
			previous_onset_preds = sum_onset_preds
			previous_midpoint_preds = sum_midpoint_preds
			previous_n_signals = sum_n_signals
			previous_n_detections = sum_n_detections
	
	total_onset_preds = previous_onset_preds
	total_midpoint_preds = previous_midpoint_preds
	total_n_signals = previous_n_signals
	total_n_detections = previous_n_detections

	total_onset_metrics = []
	total_midpoint_metrics = []
	for onset_preds, midpoint_preds in tqdm.tqdm(zip(total_onset_preds, total_midpoint_preds), total=n_grid_search):
		onset_metrics = {}
		midpoint_metrics = {}
		for c in coarsen_keys:
			onset_metrics[c] = {}
			midpoint_metrics[c] = {}
			for t in tolerance_keys:
				onset_tp, onset_fn, onset_fp = onset_preds[c][t]
				
				n = total_n_signals[c]
				
				prf1 = PRF1Metric(tolerance=t)
				try:
					onset_recall = prf1.recall(onset_tp, onset_fn)
					onset_precision = prf1.precision(onset_tp, onset_fp)
					onset_f1 = prf1.f1(onset_tp, onset_fn, onset_fp)
					onset_p_det = prf1.prob_detection(onset_tp, range(n))
					onset_p_fa = prf1.prob_false_alarm(onset_tp, onset_fp)
					onset_r_value = prf1.r_value(onset_recall, onset_precision)
				except:
					onset_recall = 0
					onset_precision = 0
					onset_f1 = 0
					onset_p_det = 0
					onset_p_fa = 0
					onset_r_value = 0

				onset_metric_dict = {
					'Recall': onset_recall,
					'Precision': onset_precision,
					'F1': onset_f1,
					'P_Detection': onset_p_det,
					'P_False_Alarm': onset_p_fa,
					'R-Value': onset_r_value
				}
				onset_metrics[c][t] = onset_metric_dict

				midpoint_tp, midpoint_fn, midpoint_fp = midpoint_preds[c][t]
				
				prf1 = PRF1Metric(tolerance=t)
				try:
					midpoint_recall = prf1.recall(midpoint_tp, midpoint_fn)
					midpoint_precision = prf1.precision(midpoint_tp, midpoint_fp)
					midpoint_f1 = prf1.f1(midpoint_tp, midpoint_fn, midpoint_fp)
					midpoint_p_det = prf1.prob_detection(midpoint_tp, range(n))
					midpoint_p_fa = prf1.prob_false_alarm(midpoint_tp, midpoint_fp)
					midpoint_r_value = prf1.r_value(midpoint_recall, midpoint_precision)
				except:
					midpoint_recall = 0
					midpoint_precision = 0
					midpoint_f1 = 0
					midpoint_p_det = 0
					midpoint_p_fa = 0
					midpoint_r_value = 0

				midpoint_metric_dict = {
					'Recall': midpoint_recall,
					'Precision': midpoint_precision,
					'F1': midpoint_f1,
					'P_Detection': midpoint_p_det,
					'P_False_Alarm': midpoint_p_fa,
					'R-Value': midpoint_r_value
				}
				midpoint_metrics[c][t] = midpoint_metric_dict
		total_onset_metrics.append(onset_metrics)
		total_midpoint_metrics.append(midpoint_metrics)

	"""
	total_onset_metrics = []
	for onset_preds in tqdm.tqdm(total_onset_preds, total=n_grid_search):
		onset_metrics = {}
		for c in coarsen_keys:
			onset_metrics[c] = {}
			for t in tolerance_keys:
				tp, fn, fp = onset_preds[c][t]
				
				n = total_n_signals[c]
				
				prf1 = PRF1Metric(tolerance=t)
				try:
					recall = prf1.recall(tp, fn)
					precision = prf1.precision(tp, fp)
					f1 = prf1.f1(tp, fn, fp)
					p_det = prf1.prob_detection(tp, range(n))
					p_fa = prf1.prob_false_alarm(tp, fp)
					r_value = prf1.r_value(recall, precision)
				except:
					recall = 0
					precision = 0
					f1 = 0
					p_det = 0
					p_fa = 0
					r_value = 0

				metric_dict = {
					'Recall': recall,
					'Precision': precision,
					'F1': f1,
					'P_Detection': p_det,
					'P_False_Alarm': p_fa,
					'R-Value': r_value
				}
				onset_metrics[c][t] = metric_dict
		total_onset_metrics.append(onset_metrics)
	"""

	results_table = p
	results_table = results_table.drop('N_signals', axis=1, inplace=False)
	results_table = results_table.drop('N_detections', axis=1, inplace=False)
	results_table = results_table.drop('OnsetPreds{coarsen:{tol:(TP,FN,FP)}}', axis=1, inplace=False)
	results_table = results_table.drop('MidpointPreds{coarsen:{tol:(TP,FN,FP)}}', axis=1, inplace=False)
	results_table = results_table.assign(N_signals=[total_n_signals] * n_grid_search)
	results_table = results_table.assign(N_detections=total_n_detections)
	results_table = results_table.assign(OnsetMetrics=total_onset_metrics)
	results_table = results_table.assign(MidpointMetrics=total_midpoint_metrics)

	results_table.to_pickle(f'{save_dir}/Results/{energy}{hpf}results_ckpt{ckpt}_{search}{q}.pkl')