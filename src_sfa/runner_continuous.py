import os
import itertools
import numpy as np
import ot

import torch
from torch.distributions import Normal
import torch.optim as optim
from utils import *
from sklearn.manifold import TSNE
from Scheduler import WarmUpScheduler
import torch.nn as nn
from torch.distributions import Normal
from sfa_llk import LLK
from sfa import FlowMatchingLoss, CNF, GaussianPrior
from callbacks import PlotLogLikelihoodCallback, PlotLossCallback

from torchdiffeq import odeint_adjoint as odeint
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from vendi_score import vendi
kernel = lambda a, b: np.dot(a,b)/100

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloader.dataloader_mnist import *
from dataloader.dataloader_pinwheel import *
from dataloader.dataloader_hvg import *

import gc
gc.collect()

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=3)



class LightningModule(pl.LightningModule):
	def __init__(self, vt: nn.Module, rt: nn.Module, priorz, config, args):
		super().__init__()
		self.config = config
		self.args = args

		self.vt = vt
		self.rt = rt

		self.priorz = priorz
		self.d = self.config.flow.z_dim
		self.p = self.config.data.size

		self.automatic_optimization = False
		self.last_validation_batch = None

	def setup(self, stage=None):

		if self.config.model.cnn:
			self.c = self.config.data.channel
			self.priory = Normal(torch.zeros(self.c, self.p, self.p).to(self.device), torch.ones(self.c, self.p, self.p).to(self.device))
		else:
			self.priory = Normal(torch.zeros(self.p), torch.ones(self.p))

		self.flow_matching_loss = FlowMatchingLoss(
			self.vt, self.rt, self.priorz,
			fixz=self.config.flow.fix_z,
			alpha=getattr(self.config.training, 'alpha', 0.001))

	def training_step(self, batch, batch_idx):
		# print("train")
		X, y = batch  # Assuming the batch is the input data `x`
		loss = self.flow_matching_loss(X.to(torch.float64))

		self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

		opt = self.optimizers()
		opt.zero_grad()

		self.manual_backward(loss)
		opt.step()

		self.clip_gradients(opt, gradient_clip_val=self.config.training.clipval, gradient_clip_algorithm="norm")

		return loss

	def validation_step(self, batch, batch_idx):
		X, y = batch
		val_loss = self.flow_matching_loss(X.to(torch.float64))

		# Store the last batch for plotting
		if batch_idx == self.trainer.num_val_batches[0] - 1:
			self.last_validation_batch = {"X": X.to(torch.float64), "y": y}

		self.log('val_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

		return val_loss

	def on_train_epoch_end(self):
		# manual scheduler step
		sch = self.lr_schedulers()
		sch.step()
	    

	def on_validation_epoch_end(self):
		# print(self.last_validation_batch)
		if self.last_validation_batch is not None:
			X = self.last_validation_batch["X"]
			y = self.last_validation_batch["y"]

			if self.current_epoch % self.config.training.snapshot_freq == 0:
				if self.args.config == "mnist.yml":
					self.generate_img(X, y, latent=True)
				else:
					# self.generate()
					self.plot_latent(X, y)

			log_post_z = self.sample_and_log(X.to(torch.float64), y)
			self.log('tra_log_post_z', log_post_z, on_step=False, on_epoch=True, sync_dist=True, logger=True)
		# Clear the stored batch for next epoch
		self.last_validation_batch = None
		
		
	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(
			itertools.chain(
			    self.vt.parameters(),
			    self.rt.parameters(),
			    ),
			lr=self.config.optim.lr,
			weight_decay=self.config.optim.weight_decay)
		cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
		                optimizer = optimizer,
		                T_max = self.config.training.n_epochs,
		                eta_min = 0,
		                last_epoch = -1
		            )
		# for group in optimizer.param_groups:
		# 	group.setdefault('initial_lr', group['lr'])
		warmUpScheduler = WarmUpScheduler(
			optimizer = optimizer,
			lr_scheduler=cosineScheduler,
			warmup_steps=5,
			warmup_start_lr=0.00001,
			len_loader=self.config.data.samplesize//self.config.training.batch_size
			)

		return [optimizer], [
			{'scheduler': warmUpScheduler,
			'monitor': 'train_loss',
			"interval":"epoch",
			"frequency":1}]


	def generate(self):
		# generate posterior predictive
		self.vt.eval()
		self.rt.eval()

		z0 = self.priorz.sample((self.config.sample.n_gen,)).to(self.config.device)
		x0 = torch.randn(self.config.sample.n_gen, self.config.data.size, device=self.config.device)
		
		x1 = self.vt.decode(x0, z0)
		z1 = self.rt.decode(z0, x1)


		xnp = x1.cpu().detach().numpy()
		# print(x1.shape)
		znp = z1.cpu().detach().numpy()


		if self.config.flow.z_dim > 1:
			# get the projection of z onto its first eigenvector direction
			znp = first_eigen_proj(znp)
		else:
			znp = znp.squeeze()

		plt.figure()
		plt.scatter(xnp[:,0], xnp[:,1], c=znp, marker=".", cmap=plt.colormaps['gist_rainbow'])
		plt.colorbar()
		plt.savefig(os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(self.current_epoch)))
		plt.close()


		self.vt.train()
		self.rt.train()

	def generate_img(self, x, y, latent=False):
		self.vt.eval()
		self.rt.eval()
		ssim = StructuralSimilarityIndexMeasure()
		loss_ssim = []

		fig, axes = plt.subplots(self.config.data.n_classes, 8, figsize=(10, 10))
		for row_idx, row_axes in enumerate(axes):
			m = len(row_axes)
			mask = y==row_idx
			if mask.sum() == 0:
			    pass
			else:
				x_k = x[mask][0]
				if self.config.model.cnn:
				    x0 = x_k.repeat(m-1,1,1,1)
				else:
				    x0 = x_k.repeat(m-1,1)
				z1 = self.priorz.sample((m-1,)).to(self.device)
				if self.config.flow.fix_z:
					z0 = self.rt.sample(x0.flatten(start_dim=1))
				else:
					z0 = self.rt.decode(z1, x0.flatten(start_dim=1))
				x1_new = self.priory.sample((m-1,)).to(self.device)
				x0_new = self.vt.decode(x1_new, z0)

				x0_new = inv_transform(x0_new)
				x0_np = x0_new.cpu().detach().numpy()
				x0 = inv_transform(x0)
				x_np = inv_transform(x_k).cpu().detach().numpy()

				if not self.config.model.cnn:
				    x0_np = x0_np.reshape((-1,self.c,self.p,self.p))
				    x_np = x_np.reshape((self.c,self.p,self.p))
			    
			for col_idx, ax in enumerate(row_axes):   
				if col_idx == 0:
					x_np_tr = np.transpose(x_np, (1, 2, 0))
					ax.imshow(x_np_tr, cmap='gray')
					ax.set_ylabel("y={}".format(row_idx))
				else:
					x0_np_tr = np.transpose(x0_np[col_idx-1], (1, 2, 0))
					ax.imshow(x0_np_tr, cmap='gray')
				ax.axis("off")

			real_img = x0.repeat(1, 3, 1, 1).to(torch.float32)
			gen_img = x0_new.repeat(1, 3, 1, 1).to(torch.float32)
			ssim_score = ssim(real_img, gen_img)

			loss_ssim.append(ssim_score)

		plt.tight_layout()
		# plt.savefig(os.path.join(self.args.log_sample_path, '{}_sampels.png'.format(ckpt_file)))
		plt.savefig(os.path.join(self.args.log_sample_path, f'image_grid_epoch_{self.current_epoch}.png'))
		plt.close()

		if latent:
			# latent 
			z1 = self.priorz.sample((len(x),)).to(self.device)
			if self.config.flow.fix_z:
				z0 = self.rt.sample(x.flatten(start_dim=1))
			else:
				z0 = self.rt.decode(z1, x.flatten(start_dim=1))
			z0_np = z0.cpu().detach().numpy()
			y_np = y.cpu().detach().numpy()
			cmap = plt.colormaps['tab10']

			tsne = TSNE(n_components=3, perplexity=30, random_state=0)
			z0_proj = tsne.fit_transform(z0_np)

			fig = plt.figure(figsize=(8, 6))
			ax = fig.add_subplot(111, projection='3d')
			scatter = ax.scatter(z0_proj[:,0], z0_proj[:,1], z0_proj[:,2], c=y_np, cmap=cmap, s=10)
			cbar = plt.colorbar(scatter, ax=ax, pad=0.1, orientation='vertical', shrink=0.5)
			ax.set_title("Generated latent z given x")

			plt.tight_layout()
			plt.savefig(os.path.join(self.args.log_sample_path, f'postz_grid_epoch_{self.current_epoch}.png'))
			plt.close()


		self.vt.train()
		self.rt.train()
		return np.array(loss_ssim).mean()


	def sample_and_log(self, x, y):
		self.vt.eval()
		self.rt.eval()
		# posterior log lik

		z1 = self.priorz.sample((len(x),)).to(self.device)
		if self.config.flow.fix_z:
			z0 = self.rt.sample(x.flatten(start_dim=1))
			log_post_z = self.rt.log_prob(x.flatten(start_dim=1), z0).mean()
		else:
			z0 = self.rt.decode(z1, x.flatten(start_dim=1))
			log_post_z = self.rt.log_prob(z0, x.flatten(start_dim=1), 0, self.priorz).mean()
		# log_lik = self.vt.log_prob(x, z0, 0, self.priory).mean()

		self.vt.train()
		self.rt.train()

		return log_post_z #, log_lik

	def test_generate(self, n):
		z1 = self.priorz.sample((n,)).to(self.config.device)
		x1 = torch.randn(n, self.config.data.size, device=self.config.device)

		x0 = self.vt.decode(x1, z1)
		if self.config.flow.fix_z:
			z0 = self.rt.sample(x0)
		else:
			z1 = self.priorz.sample((len(x),)).to(self.config.device)
			z0 = self.rt.decode(z1, x0)

		return x0

	def test_regenerate(self, x, y):
		
		x1 = torch.randn(len(x), self.config.data.size, device=self.config.device)
		if self.config.flow.fix_z:
			z0 = self.rt.sample(x)
		else:
			z1 = self.priorz.sample((len(x),)).to(self.config.device)
			z0 = self.rt.decode(z1, x)

		x0 = self.vt.decode(x1, z0)
		x0_np = x0.cpu().detach().numpy()
		y0_np = y.cpu().detach().numpy()

		# kernel = lambda a, b: np.exp(-np.linalg.norm(a - b)**2/2)
		# vs = vendi.score(x0_np, kernel)
		vs = vendi.score_dual(x0_np, normalize=True)

		# plot the umap
		dict = {'CD14 Mono': 0, 'CD4 T': 1, 'T': 2, 'CD8 T': 3, 'B': 4, 'DC': 5, 'CD16 Mono': 6, 'NK': 7}
		inv_dict = {v: k for k, v in dict.items()}
		cell_type = [inv_dict.get(i) for i in y0_np]

		adata0 = ad.AnnData(x0_np)
		adata0.obs['cell_type'] = cell_type
		
		return vs, x0_np

	def plot_latent(self, x, y):
		# latent 
		z1 = self.priorz.sample((len(x),)).to(self.device)
		if self.config.flow.fix_z:
			z0 = self.rt.sample(x.flatten(start_dim=1))
		else:
			z0 = self.rt.decode(z1, x.flatten(start_dim=1))
		z0_np = z0.cpu().detach().numpy()
		y_np = y.cpu().detach().numpy()
		
		cmap = plt.colormaps['tab10']
		if z1.shape[1] == 1:
			fig = plt.figure(figsize=(8, 6))
			ax = fig.add_subplot(111)
			scatter = ax.scatter(x[:,0], x[:,1], c=z0_np, cmap=plt.colormaps['hsv'], s=10)
			cbar = plt.colorbar(scatter, ax=ax, pad=0.1, orientation='vertical', shrink=0.5)
			# ax.set_title("Generated latent z given x")

			plt.tight_layout()
			plt.savefig(os.path.join(self.args.log_sample_path, f'test_latent_epoch_{self.current_epoch}.png'))
			plt.close()
		else:
			tsne = TSNE(n_components=2, perplexity=50, random_state=0) # perplexity=30 for mnist
			z0_proj = tsne.fit_transform(z0_np)

			fig = plt.figure(figsize=(8, 6))
			ax = fig.add_subplot(111)
			scatter = ax.scatter(z0_proj[:,0], z0_proj[:,1], c=y_np, cmap=cmap, s=10)
			cbar = plt.colorbar(scatter, ax=ax, pad=0.1, orientation='vertical', shrink=0.5)
			# ax.set_title("Generated latent z given x")

			plt.tight_layout()
			plt.savefig(os.path.join(self.args.log_sample_path, f'test_latent_epoch_{self.current_epoch}.png'))
			plt.close()


	def test_plot_latent(self, x, y, in_sample=True):
		if in_sample:
			name = "mnist"
			k = self.config.data.n_classes
		else:
			name = "emnist"
			k = 20
		# latent 
		z1 = self.priorz.sample((len(x),)).to(self.device)
		if self.config.flow.fix_z:
			z0 = self.rt.sample(x.flatten(start_dim=1))
		else:
			z0 = self.rt.decode(z1, x.flatten(start_dim=1))
		z0_np = z0.cpu().detach().numpy()
		y_np = y.cpu().detach().numpy()
		
		cmap = plt.colormaps['tab20']

		tsne = TSNE(n_components=2, perplexity=50, random_state=0) # perplexity=30 for mnist
		z0_proj = tsne.fit_transform(z0_np)

		fig = plt.figure(figsize=(8, 6))
		ax = fig.add_subplot(111)
		scatter = ax.scatter(z0_proj[:,0], z0_proj[:,1], c=y_np, cmap=cmap, s=10)
		cbar = plt.colorbar(scatter, ax=ax, pad=0.1, orientation='vertical', shrink=0.5)
		# ax.set_title("Generated latent z given x")

		plt.tight_layout()
		plt.savefig(os.path.join(self.args.log_sample_path, f'test_latent_{name}.png'))
		plt.close()

		if not self.training:
			vs = vendi.score_dual(z0_np, normalize=True)

			# clustering
			kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
			kmeans.fit(z0_np)
			k0_np = kmeans.labels_

			ari = adjusted_rand_score(y_np, k0_np)
			# nmi = soft_nmi(pi_np, y_np)
			nmi = normalized_mutual_info_score(y_np, k0_np)
			# ari = soft_ari(pi_np, y_np)
			return nmi, ari, vs


	def test_ood(self, X, y):

		is_digit_mask = (y >= 0) & (y <= 9)
		x = X[is_digit_mask]
		x_ood = X[~is_digit_mask]
		# print("x", x.shape)
		# print("x_odd", x_ood.shape)

		# compute the entropy of posterior given train samples
		z1 = self.priorz.sample((len(x),)).to(self.device)
		if self.config.flow.fix_z:
			z0 = self.rt.sample(x.flatten(start_dim=1))
		else:
			z0 = self.rt.decode(z1, x.flatten(start_dim=1))

		# compute posterior entropy
		post_ent = - self.rt.log_prob(x.flatten(start_dim=1), z0).mean()

		# compute posterior of ood samples
		# compute the entropy of posterior given train samples
		z1_ood = self.priorz.sample((len(x_ood),)).to(self.device)
		if self.config.flow.fix_z:
			z0_ood = self.rt.sample(x_ood.flatten(start_dim=1))
		else:
			z0_ood = self.rt.decode(z1, x_ood.flatten(start_dim=1))
		post_ent_ood = - self.rt.log_prob(x_ood.flatten(start_dim=1), z0_ood).mean()

		# sba difference
		eps= torch.abs(post_ent - post_ent_ood)
		# eps= post_ent - post_ent_ood

		return eps
		
	def on_test_epoch_start(self):
		self.eps = []
	
	def on_test_epoch_end(self):
		outputs = self.eps
		print(f"test_epoch_end received {len(outputs)} items (batches).")
		if not outputs:
			print("No outputs from test_step to process for histogram in test_epoch_end.")
			return

		aggregated_values = []
		for batch_output in outputs:
			if isinstance(batch_output, torch.Tensor):
			    aggregated_values.extend(batch_output.detach().cpu().numpy().flatten().tolist())
			elif isinstance(batch_output, (list, np.ndarray)):
			     aggregated_values.extend(np.array(batch_output).flatten().tolist())
			else:
			    aggregated_values.append(batch_output)

		self.eps.extend(aggregated_values)


	def test_step(self, batch, batch_idx):
		X, y = batch  # Assuming the batch is the input data `x`
		# loss = self.flow_matching_loss(X.to(torch.float64))
		n = len(X)

		# compute distance between true sample to generated samples
		if self.config.model.cnn:
			if self.args.sample:
				nmi, ari, vs = self.test_plot_latent(X.to(torch.float64), y, in_sample=self.config.data.in_sample)
				self.log('test_nmi', nmi, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
				self.log('test_ari', ari, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
				self.log('test_vendi', vs, on_step=True, on_epoch=True, sync_dist=True, logger=True)

			elif self.args.ood:
				eps = self.test_ood(X.to(torch.float64), y)
				if eps is not None:
					self.eps.append(eps)
		else:
			if self.args.config == "pinwheel.yml":

				xnew = self.test_generate(n)
				xnew = xnew.cpu().detach().numpy()

				xorg = X.cpu().detach().numpy()

				w = 1/n * np.ones(n)

				M = ot.dist(xorg, xnew, "euclidean")
				M /= M.max() * 0.1
				d_emd = ot.emd2(w, w, M)

				self.log('test_emd', d_emd, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
			elif self.args.config == "hvg.yml":
				nmi, ari, vs = self.test_plot_latent(X.to(torch.float64), y)
				self.log('test_nmi', nmi, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
				self.log('test_ari', ari, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
				self.log('test_vendi', vs, on_step=True, on_epoch=True, sync_dist=True, logger=True)
				# also evaluate the generated x 
				vs_x, xnew = self.test_regenerate(X.to(torch.float64), y)
				self.log('test_vendi_x', vs_x, on_step=True, on_epoch=True, sync_dist=True, logger=True)

				# xnew = self.test_generate(n)
				# xnew = xnew.cpu().detach().numpy()

				xorg = X.cpu().detach().numpy()

				w = 1/n * np.ones(n)

				M = ot.dist(xorg, xnew, "euclidean")
				M /= M.max() * 0.1
				d_emd = ot.emd2(w, w, M)

				self.log('test_emd', d_emd, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)


class PinWheelRunner():
	def __init__(self, args, config):
		self.args = args
		self.config = config
		args.log_sample_path = os.path.join(args.log_path, 'samples')
		os.makedirs(args.log_sample_path, exist_ok=True)

		# print(self.args)
		self.d = self.config.flow.z_dim
		self.p = self.config.data.size

		self.vt = LLK(self.config.data.size, self.config.flow.z_dim, model="mlp", hidden_features=[self.config.model.ngf]*5).to(self.config.device)
		if self.config.flow.fix_z:
			self.rt = GaussianPrior(
				self.config.data.size, self.d, hidden_features=[self.config.flow.ngf]*1,
				freqs=self.config.flow.freqs,
				fct=nn.Tanh(),
				).to(self.config.device)
		else:
			self.rt = CNF(
				self.config.data.size, 
				self.config.flow.z_dim, 
				hidden_features=[self.config.flow.ngf]*0,
				freqs=self.config.flow.freqs
				).to(self.config.device)

		self.priorz = Normal(torch.zeros(self.d), torch.ones(self.d))

		# Define the ModelCheckpoint callback
		checkpoint_callback = ModelCheckpoint(
		    monitor='val_loss',  # Metric to monitor
		    dirpath=self.args.log_path,  # Directory where checkpoints will be saved
		    filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',  # Filename convention
		    save_top_k=5,  # Only save the best model based on val_loss
		    mode='min'  # Minimize the validation loss
		)
		plot_llk_callback = PlotLogLikelihoodCallback(save_path=self.args.log_sample_path, log_keys=("tra_log_lik", "tra_log_post_z"))
		plot_loss_callback = PlotLossCallback(save_path=os.path.join(self.args.log_sample_path, f'loss.png'), update_interval=1)
		# Initialize the Trainer

		if torch.cuda.is_available():
			accelerator='gpu'
			strategy="ddp"
			devices="auto"
		else:
			accelerator='cpu'
			devices="auto"
			strategy = "auto"

		self.trainer = pl.Trainer(
			max_epochs=self.config.training.n_epochs, 
			# accelerator='gpu',
			accelerator = accelerator,
			devices=devices,
			strategy=strategy,
			callbacks=[checkpoint_callback, plot_llk_callback, plot_loss_callback]
		)

	def train(self):
		# load data
		dataset, test_dataset = get_dataset(self.config.data.n_classes, "data", self.config.data.samplesize, self.config.data.test_samplesize)
		train_dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
								num_workers=self.config.data.num_workers)
		val_dataloader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
									num_workers=self.config.data.num_workers, drop_last=True)

		model = LightningModule(self.vt, self.rt, self.priorz, self.config, self.args)
		# Run the training loop
		if not self.args.resume_training:
			ckpt_path = None
		else:
			_ckpt_path = os.path.join(self.args.tb_path, "lightning_logs/version_0/checkpoints")
			ckpt_files = [f for f in 
					os.listdir(_ckpt_path) if f.endswith('.ckpt')]
			ckpt_path = os.path.join(_ckpt_path, ckpt_files[-1])

		self.trainer.fit(
			model, train_dataloader, val_dataloader, ckpt_path=ckpt_path
			)

	def sample(self):
		dataset, test_dataset = get_dataset(self.config.data.n_classes, "data", 2000, 2000)

		test_dataloader = DataLoader(test_dataset, batch_size=500, shuffle=True,
							num_workers=self.config.data.num_workers, drop_last=True)

		model = LightningModule(self.vt, self.rt, self.priorz, self.config, self.args)

		_ckpt_path = os.path.join(self.args.tb_path, "lightning_logs/version_0/checkpoints")
		ckpt_files = [f for f in 
		os.listdir(_ckpt_path) if f.endswith('.ckpt')]
		ckpt_path = os.path.join(_ckpt_path, ckpt_files[-1])

		self.trainer.test(model, dataloaders=test_dataloader, ckpt_path=ckpt_path)

class MNISTRunner():
	def __init__(self, args, config):
		self.args = args
		self.config = config
		args.log_sample_path = os.path.join(args.log_path, 'samples')
		os.makedirs(args.log_sample_path, exist_ok=True)
		# print(self.args)
		self.d = self.config.flow.z_dim
		self.c, self.p = self.config.data.channel, self.config.data.size

		self.priorz = Normal(torch.zeros(self.d), torch.ones(self.d))

		vt_type = getattr(self.config.model, 'type', 'cnn')
		self.vt = LLK(
			self.p, self.d, model=vt_type, is_image=True, in_ch=self.c,
			freqs=self.config.model.freqs,
			hidden_features=[self.config.flow.ngf]*3,
			fct=nn.SiLU(),
			# fct=nn.Tanh(),
			# layer_norm=True,
			# fct=nn.ReLU(),
			# fct=nn.Softplus(),
			hidden_dim=28*28,
			)

		if self.config.flow.fix_z:
			self.rt = GaussianPrior(
				self.c*self.p**2, self.d, hidden_features=[self.config.flow.ngf]*0,
				freqs=self.config.flow.freqs,
				fct=nn.Tanh(),
				).to(self.config.device)
		else:
			self.rt = CNF(
			    self.c*self.p**2, self.d, hidden_features=[self.config.flow.ngf]*0,
			    hidden_dim=256,
			    freqs=self.config.flow.freqs,
			    ).to(self.config.device)


		# Define the ModelCheckpoint callback
		self.checkpoint_callback = ModelCheckpoint(
			monitor='val_loss',  # Metric to monitor
			dirpath=self.args.log_path,  # Directory where checkpoints will be saved
			filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',  # Filename convention
			save_top_k=1,  # Only save the best model based on val_loss
			mode='min'  # Minimize the validation loss
		)
		# Initialize the Trainer

		if torch.cuda.is_available():
			accelerator='gpu'
			strategy="ddp"
			devices="auto"
		else:
			accelerator='cpu'
			devices="auto"
			strategy = "auto"

		plot_loss_callback = PlotLossCallback(save_path=os.path.join(self.args.log_sample_path, f'loss.png'), update_interval=1)
		plot_llk_callback = PlotLogLikelihoodCallback(save_path=self.args.log_sample_path, log_keys=("tra_log_lik", "tra_log_post_z"))

		self.trainer = pl.Trainer(
			max_epochs=self.config.training.n_epochs, 
			# accelerator='gpu',
			accelerator = accelerator,
			devices=devices,
			strategy=strategy,
			callbacks=[plot_loss_callback, plot_llk_callback, self.checkpoint_callback]
		)

	def train(self):
		# load data
		dataset, val_dataset, sampler, val_sampler = get_mnist(
			self.config.data.n_classes, "data", self.config.data.samplesize, self.config.data.test_samplesize)
		train_dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size,
		                    num_workers=self.config.data.num_workers, sampler=sampler)
		val_dataloader = DataLoader(val_dataset, batch_size=self.config.training.batch_size, 
		                     num_workers=self.config.data.num_workers, sampler=val_sampler, drop_last=True)
		# Initialize the Lightning model
		model = LightningModule(self.vt, self.rt, self.priorz, self.config, self.args)
		# Run the training loop
		if not self.args.resume_training:
			ckpt_path = None
		else:
			ckpt_path = self.checkpoint_callback.best_chechpoint_callback

		self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)

	def sample(self):
		if self.config.data.in_sample:
			dataset, _, sampler, _ = get_mnist(
				self.config.data.n_classes, "data", 300, 50)
			dataloader = DataLoader(dataset, batch_size=3000,
			                    num_workers=self.config.data.num_workers, sampler=sampler)

		else:
			# 0123456789ABCDEFGHIJ
			dataset, sampler = get_emnist(
				20, "data", 300, 50, split="balanced") # balanced
			dataloader = DataLoader(dataset, batch_size=6000,
			                    num_workers=self.config.data.num_workers, sampler=sampler)
		model = LightningModule(self.vt, self.rt, self.priorz, self.config, self.args)

		
		ckpt_path = self.checkpoint_callback.best_chechpoint_callback

		self.trainer.test(model, dataloaders=dataloader, ckpt_path=ckpt_path)

	def ood(self):
		# 0123456789ABCDEFGHIJ
		dataset, sampler = get_emnist(
				20, "data", 3000, 500, split="balanced", replace=True) # balanced
		dataloader = DataLoader(dataset, batch_size=200,
		                    num_workers=self.config.data.num_workers, sampler=sampler)
		model = LightningModule(self.vt, self.rt, self.priorz, self.config, self.args)

		ckpt_path = self.checkpoint_callback.best_chechpoint_callback

		self.trainer.test(model, dataloaders=dataloader, ckpt_path=ckpt_path)




class HVGRunner():
	def __init__(self, args, config):
		self.args = args
		self.config = config
		args.log_sample_path = os.path.join(args.log_path, 'samples')
		os.makedirs(args.log_sample_path, exist_ok=True)


		# print(self.args)
		self.d = self.config.flow.z_dim
		self.p = self.config.data.size

		self.vt = LLK(
					self.p,
					self.d,
					model="mlp_high",
					hidden_features=[self.config.model.ngf]*3,
					hidden_dim=800, # 500 was ok, 800 is unstable
					fct=nn.Softplus(),
					freqs=20,
					).to(self.config.device)

		self.rt = self.rt = GaussianPrior(
					self.p, self.d, hidden_features=[self.config.flow.ngf]*0,
					freqs=2,
					fct=nn.Softplus(),
					).to(self.config.device) 


		self.priorz = Normal(torch.zeros(self.d), torch.ones(self.d))

		# Define the ModelCheckpoint callback
		self.checkpoint_callback = ModelCheckpoint(
		    monitor='val_loss',  # Metric to monitor
		    dirpath=self.args.log_path,  # Directory where checkpoints will be saved
		    filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',  # Filename convention
		    save_top_k=1,  # Only save the best model based on val_loss
		    mode='min'  # Minimize the validation loss
		)
		# Initialize the Trainer

		if torch.cuda.is_available():
			accelerator='gpu'
			strategy="ddp"
			devices="auto"
		else:
			accelerator='cpu'
			devices="auto"
			strategy = "auto"

		plot_loss_callback = PlotLossCallback(save_path=os.path.join(self.args.log_sample_path, f'loss.png'), update_interval=1)
		plot_llk_callback = PlotLogLikelihoodCallback(save_path=self.args.log_sample_path, log_keys=("tra_log_lik", "tra_log_post_z"))

		self.trainer = pl.Trainer(
			max_epochs=self.config.training.n_epochs, 
			# accelerator='gpu',
			accelerator = accelerator,
			devices=devices,
			strategy=strategy,
			callbacks=[plot_loss_callback, plot_llk_callback, self.checkpoint_callback]
		)


	def train(self):
		# load data
		train_dataloader, val_dataloader, _ = get_hvg_dataloaders(
			dat_dir="data", batch_size= self.config.training.batch_size,
			num_workers=self.config.data.num_workers)

		model = LightningModule(self.vt, self.rt, self.priorz, self.config, self.args)
		# Run the training loop
		if not self.args.resume_training:
			ckpt_path = None
		else:
			ckpt_path = self.checkpoint_callback.best_model_path

		self.trainer.fit(
			model, train_dataloader, val_dataloader, ckpt_path=ckpt_path
			)

	def sample(self):
		_, val_dataloader, test_dataloader = get_hvg_dataloaders(
	    	dat_dir="data", batch_size=1226, num_workers=self.config.data.num_workers)
		model = LightningModule(self.vt, self.rt, self.priorz, self.config, self.args)

		ckpt_path = self.checkpoint_callback.best_model_path

		self.trainer.test(model, dataloaders=val_dataloader, ckpt_path=ckpt_path)
