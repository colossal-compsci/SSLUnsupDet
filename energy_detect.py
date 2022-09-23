import re 
import os
import glob
import json
import argparse
import pandas as pd
from scipy.signal import find_peaks

from metrics import *
from utils import *
from layers import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-c', '--config', type=str,
						help='JSON file for configuration')
	parser.add_argument('-q', '--use_q', type=int,
						help='use uncertain (q) selection tables (0 or 1)',
						default=1)
	parser.add_argument('-hpf', '--use_hpf', type=int,
						help='use high pass filter (0 or 1)',
						default=0)
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
	base_path = config['dataset']['wavs_path'][:-6]

	ckpt = args.checkpoint
	search = args.search
	assert search in ['coarse', 'fine']

	prominence = range_step(config['energy_baseline'][search]['prominence'])
	sample_rate = config['dataset']['sample_rate']
	tolerance = config['metrics']['tolerance']
	coarsen = config['metrics']['coarsen']

	use_hpf = True if args.use_hpf == 1 else False
	if use_hpf:
		hpf = HighPassFilter(**config['model']['preprocess_params']['filter_params'])
		hpf_str = 'hpf_'
	else:
		hpf_str = '' 

	prf1metrics = [PRF1Metric(tolerance=t) for t in tolerance]

	inference_wavs = sorted(glob.glob(f'{save_dir}/Inference/*/predictions_ckpt{ckpt}_{search}*.pkl'))
	inference_wavs = [re.findall('\d+\w+', w)[0] for w in inference_wavs ]

	for wav in tqdm.tqdm(inference_wavs):
		prominences = []
		N_signals = []
		N_detections = []
		predictions_wrt_onsets = []
		predictions_wrt_midpoints = []

		if not use_q:
			if os.path.exists(f'data/selections/{wav}.selections.txt'):
				selections_table = pd.read_csv(f'data/selections/{wav}.selections.txt', sep='\t')
				q = ""
			else:
				selections_table = None
		else:
			if os.path.exists(f'data/selections/{wav}.selections.txt'):
				selections_table = pd.read_csv(f'data/selections/{wav}.selections.txt', sep='\t')
				q = ""
			elif os.path.exists(f'data/selections/{wav}.q.selections.txt'):
				selections_table = pd.read_csv(f'data/selections/{wav}.q.selections.txt', sep='\t')
				q = r".q"
			else:
				selections_table = None

		if selections_table is not None:
			wav_save_path = f'{save_dir}/Inference/{wav}/energy_{hpf_str}predictions_ckpt{ckpt}_{search}{q}.pkl'
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

				x, _ = librosa.load(f'{base_path}/{wav}.wav', sr=sample_rate)
				if use_hpf:
					x = torch.tensor(x).view(1, 1, x.shape[0])
					x = hpf(x).squeeze().numpy()
				energy = x ** 2

				for p in prominence:
					pks, _ = find_peaks(energy, prominence=p)

					detections = sample_to_time(np.array(pks), sample_rate)
					N_detection = len(detections)

					preds_wrt_onsets = {c:{prf1.tolerance:prf1.classify_predictions(onset_times, detections) for prf1 in prf1metrics} for c, onset_times in onsets.items()}
					preds_wrt_midpoints = {c:{prf1.tolerance:prf1.classify_predictions(midpoint_times, detections) for prf1 in prf1metrics} for c, midpoint_times in midpoints.items()}

					prominences.append(p)
					N_signals.append(N_signal)
					N_detections.append(N_detection)
					predictions_wrt_onsets.append(preds_wrt_onsets)
					predictions_wrt_midpoints.append(preds_wrt_midpoints)
				
				df = pd.DataFrame({
						'Prominences': prominences,
						'N_signals': N_signals,
						'N_detections': N_detections,
						'OnsetPreds{coarsen:{tol:(TP,FN,FP)}}': predictions_wrt_onsets,
						'MidpointPreds{coarsen:{tol:(TP,FN,FP)}}': predictions_wrt_midpoints
						})

				df.to_pickle(wav_save_path)