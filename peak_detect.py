import re
import os
import glob
import json
import tqdm
import argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from utils import *
from metrics import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-c', '--config', type=str,
						help='JSON file for configuration')
	parser.add_argument('-q', '--use_q', type=int,
						help='use uncertain (q) selection tables (0 or 1)',
						default=1)
	parser.add_argument('-ckpt', '--checkpoint', type=int,
						help='choose the model ckpt')
	parser.add_argument('-s', '--search', type=str,
						help='choose the search regime (e.g. coarse or fine)')

	args = parser.parse_args()

	with open(f'configs/{args.config}') as f:
		data = f.read()
	f.close()
	config = json.loads(data)

	save_dir = config['utils']['save_dir']
	use_q = True if args.use_q == 1 else False
	
	ckpt = args.checkpoint
	search = args.search
	assert search in ['coarse', 'fine']

	prominence = range_step(config['detection'][search]['prominence'])
	hop_length = config['model']['transform_params']['params']['stride']
	sample_rate = config['dataset']['sample_rate']
	tolerance = config['metrics']['tolerance']
	coarsen = config['metrics']['coarsen']

	

	prf1metrics = [PRF1Metric(tolerance=t) for t in tolerance]

	wav_distances = sorted(glob.glob(f'{save_dir}/Inference/*/distances_ckpt{ckpt}_{search}.pkl'))
	assert len(wav_distances) > 0

	for wav_distance in tqdm.tqdm(wav_distances):
		
		alphas = []
		betas = []
		smooth_durations = []
		prominences = []
		N_signals = []
		N_detections = []
		predictions_wrt_onsets = []
		predictions_wrt_midpoints = []
		
		filename = re.findall('\d+\w+', wav_distance)[0]

		if not use_q:
			if os.path.exists(f'data/selections/{filename}.selections.txt'):
				selections_table = pd.read_csv(f'data/selections/{filename}.selections.txt', sep='\t')
				q = ""
			else:
				selections_table = None
		else:
			if os.path.exists(f'data/selections/{filename}.selections.txt'):
				selections_table = pd.read_csv(f'data/selections/{filename}.selections.txt', sep='\t')
				q = ""
			elif os.path.exists(f'data/selections/{filename}.q.selections.txt'):
				selections_table = pd.read_csv(f'data/selections/{filename}.q.selections.txt', sep='\t')
				q = r".q"
			else:
				selections_table = None
		
		if selections_table is not None:
			wav_save_path = f'{save_dir}/Inference/{filename}/predictions_ckpt{ckpt}_{search}{q}.pkl'
			
			if not os.path.exists(wav_save_path):
				assert selections_table.View.iloc[0] == 'Waveform 1'
				assert selections_table.View.iloc[1] == 'Spectrogram 1'
				selections_table = selections_table[::2].reset_index(drop=True)
				
				begin_times = selections_table['Begin Time (s)'].to_numpy()
				end_times = selections_table['End Time (s)'].to_numpy()
				
				onsets = {0: begin_times}
				offsets = {0: end_times}
				for c in coarsen:
					onsets[c] = coarsen_selections(begin_times, 
												   end_times, 
												   threshold=c)[0]
					offsets[c] = coarsen_selections(begin_times, 
													end_times, 
													threshold=c)[1]

				midpoints = {}
				for k in onsets.keys():
					midpoints[k] = (onsets[k] + offsets[k]) / 2

				N_signal = {k: len(v) for k, v in onsets.items()}

				distances_table = pd.read_pickle(wav_distance)
				
				for p in prominence:
					for i, r in distances_table.iterrows():

						a = np.round(r.Alphas, 2)
						b = np.round(r.Betas, 2)
						s = np.round(r.SmoothDurations, 2)
						
						distance = r.Distances

						pks, _ = find_peaks(distance, prominence=p)

						detections = frame_to_time(np.array(pks), hop_length, sample_rate)
						N_detection = len(detections)

						preds_wrt_onsets = {c:{prf1.tolerance:prf1.classify_predictions(onset_times, detections) for prf1 in prf1metrics} for c, onset_times in onsets.items()}
						preds_wrt_midpoints = {c:{prf1.tolerance:prf1.classify_predictions(midpoint_times, detections) for prf1 in prf1metrics} for c, midpoint_times in midpoints.items()}

						alphas.append(a)
						betas.append(b)
						smooth_durations.append(s)
						prominences.append(p)
						N_signals.append(N_signal)
						N_detections.append(N_detection)
						predictions_wrt_onsets.append(preds_wrt_onsets)
						predictions_wrt_midpoints.append(preds_wrt_midpoints)
						
				df = pd.DataFrame({
					'Alphas': alphas,
					'Betas': betas,
					'SmoothDurations': smooth_durations,
					'Prominences': prominences,
					'N_signals': N_signals,
					'N_detections': N_detections,
					'OnsetPreds{coarsen:{tol:(TP,FN,FP)}}': predictions_wrt_onsets,
					'MidpointPreds{coarsen:{tol:(TP,FN,FP)}}': predictions_wrt_midpoints
					})

				df.to_pickle(wav_save_path)