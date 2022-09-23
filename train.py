import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import json
import argparse

from dataset import *
from models import *
from metrics import *
from losses import *
from utils import *
from PyFire import *

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	
	parser.add_argument('-c', '--config', type=str,
						help='JSON file for configuration')
	parser.add_argument('-s', '--seed', type=int,
						help='choose the seed', default=42)

	args = parser.parse_args()
	config = args.config
	seed = args.seed

	seed_everything(seed)

	with open(f'configs/{args.config}') as f:
		data = f.read()
	f.close()
	config = json.loads(data)

	save_dir = config['utils']['save_dir']

	dataset_params = config['dataset']
	training_params = config['training']
	model_params = config['model']

	train_set = SpermWhaleClicks(n_samples=dataset_params['train_samples'], 
								 base_path=dataset_params['wavs_path'],
								 subset='train',
								 window=dataset_params['window'],
								 window_pad=dataset_params['window_pad'],
								 sample_rate=dataset_params['sample_rate'],
								 epsilon=dataset_params['epsilon'],
								 seed=dataset_params['seed'])
	val_set = SpermWhaleClicks(n_samples=dataset_params['val_samples'], 
							   base_path=dataset_params['wavs_path'],
							   subset='val',
							   window=dataset_params['window'],
							   window_pad=dataset_params['window_pad'],
							   sample_rate=dataset_params['sample_rate'],
							   epsilon=dataset_params['epsilon'],
							   seed=dataset_params['seed'])  

	train_loader = DataLoader(train_set, 
							  batch_size=training_params['batch_size'], 
							  shuffle=True)
	val_loader = DataLoader(val_set,
							batch_size=training_params['batch_size'], 
							shuffle=False)

	model = SpectralBoundaryEncoder(**model_params)

	if training_params['optimizer']['name'] == 'SGD':
		optimizer = optim.SGD(model.parameters(), 
							  lr=training_params['optimizer']['learning_rate'], 
							  momentum=training_params['optimizer']['momentum'])

	scheduler = training_params['scheduler']
	if scheduler is not None:
		if scheduler['type'] == 'plateau':
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **scheduler['kwargs'])
		elif scheduler['type'] == 'step':
			scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler['kwargs'])
		elif scheduler['type'] == 'multi_step':
			scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler['kwargs'])
			
	try:
		n_negatives = training_params['n_negatives']
	except KeyError:
		n_negatives = 1
	print(f'n_negatives: {n_negatives}')

	nce_loss = NoiseContrastiveEstimationLoss(n_negatives=n_negatives)
	def loss_fx(z, dummy_arg):
		preds = nce_loss.compute_preds(z)
		loss = nce_loss.loss(preds)
		return loss

	nce_metric = NoiseContrastiveEstimationMetric()
	def metric_fx(z, dummy_arg, index=1):
		preds = nce_metric.compute_preds(z)
		metric = nce_metric.metric(preds)
		return metric

	trainer = Trainer(model, 
					  optimizer,
					  scheduler=scheduler,
					  loss_func={'NE Loss': loss_fx},
					  metric_func={'NE Metric': metric_fx},
					  verbose=training_params['verbose'],
					  device=training_params['device'],
					  dest=save_dir,
					  **training_params['params'])
	trainer.fit(train_loader, val_loader, training_params['epochs'])
	trainer.save_model()