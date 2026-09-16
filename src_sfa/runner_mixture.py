import os
import itertools
import numpy as np

from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from functools import partial
import torch
from torch.distributions import Normal
import torch.optim as optim
from utils import *
from Scheduler import GradualWarmupScheduler, WarmUpScheduler
from sfa_llk import LLK
from sfa_discrete import *
from models.nn import to_one_hot
from callbacks import PlotLogLikelihoodCallback, PlotLossCallback

from dataloader.dataloader_mnist import *
from dataloader.dataloader_pinwheel import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy


import gc
gc.collect()

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=3)

class PinWheelMixtureLightningModule(pl.LightningModule):
    def __init__(self, vt: nn.Module, Rt: nn.Module, rt: nn.Module, priorz, config, args):
        super().__init__()
        self.config = config
        self.args = args
        
        self.vt = vt
        self.Rt = Rt
        self.rt = rt
        self.priorz = priorz

        self.k, self.d = self.config.flow.k_dim, self.config.flow.z_dim
        self.p = self.config.data.size
        self.temp_max = self.config.flow.beta
        self.temp_min = 0.5
        self.beta = self.temp_max
        self.alpha = self.config.training.alpha
        # Register buffers for priorz parameters

        self.automatic_optimization = False
        self.last_validation_batch = None


    def setup(self, stage=None):
        self.priorpi = Normal(torch.zeros(self.k, device=self.device), torch.ones(self.k, device=self.device))

        self.flow_matching_loss = FlowMatchingLossMixture(
            self.vt, self.Rt, self.rt, self.priorpi, self.priorz,
            k=self.k, alpha=self.alpha)

    def training_step(self, batch, batch_idx):
        # print("train")
        X, y = batch  # Assuming the batch is the input data `x`
        loss = self.flow_matching_loss(X.to(torch.float64))

        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        opt = self.optimizers()
        opt.zero_grad()

        self.manual_backward(loss)
        opt.step()

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
        if self.last_validation_batch is not None:
            X = self.last_validation_batch["X"]
            y = self.last_validation_batch["y"]

            if self.current_epoch % self.config.training.snapshot_freq == 0:

                """ Snapshot sampling at the end of every epoch """
                # if self.config.training.snapshot_sampling:
                log_post_z, log_lik = self.sample_and_log(X, y)
                self.generate(X, y)
                # self.sample_and_log()
                # self.log('tra_log_post_pi', log_post_pi, on_step=False, on_epoch=True, sync_dist=True, logger=True)
                self.log('tra_log_post_z', log_post_z, on_step=False, on_epoch=True, sync_dist=True, logger=True)
                self.log('tra_log_lik', log_lik, on_step=False, on_epoch=True, sync_dist=True, logger=True)

        # Clear the stored batch for next epoch
        self.last_validation_batch = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(
                self.vt.parameters(),
                self.rt.parameters(),
                self.Rt.parameters(),
                ),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay)

        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = self.config.training.n_epochs,
                            eta_min = 0,
                            last_epoch = -1
                        )

        warmUpScheduler = GradualWarmupScheduler(
                                optimizer = optimizer,
                                multiplier = self.config.optim.multiplier,
                                warm_epoch = 5,
                                after_scheduler = cosineScheduler,
                                last_epoch = self.current_epoch
                            )

        return [optimizer], [
                {'scheduler': warmUpScheduler,
                'monitor': 'train_loss',
                "interval":"epoch",
                "frequency":1}]

    def sample_and_log(self, x, y):
        self.vt.eval()
        self.Rt.eval()
        self.rt.eval()
        # self.priorz.eval()

        with torch.no_grad():
            # evluate log probability 
            x1 = torch.randn(len(x), self.config.data.size)
            # generated samples
            logits1 = self.priorpi.sample((len(x),)).to(x.device)
            # z1idx = F.gumbel_softmax(logits1, tau=self.config.flow.tau, hard=False)
            # z1 = self.priorz.sample(z1idx, (1,)).to(x.device)

            t0 = torch.tensor(0., device=x.device)
            logits0, z0idx = self.Rt.rsample(x, t0)
            z0 = self.rt.sample(logits0, x)
            # pi0 = softmax(logits0/self.beta, dim=-1)

            log_post_z = self.rt.log_prob(logits0, x, z0).mean()
            log_lik = self.vt.log_prob(x, z0, 0.).mean()

        self.vt.train()
        self.Rt.train()
        self.rt.train()
        # self.priorz.train()

        return log_post_z, log_lik

    def generate(self, x, y):
        self.vt.eval()
        self.Rt.eval()
        self.rt.eval()
        # self.priorz.eval()

        with torch.no_grad():
            # posterior predictive
            x1 = torch.randn(self.config.sample.n_gen, self.config.data.size)
            # generated samples
            logits1 = self.priorpi.sample((self.config.sample.n_gen,)).to(x.device)
            # z1idx = F.gumbel_softmax(logits1, tau=self.config.flow.tau, hard=False)
            t0 = torch.tensor(0., device=self.device)
            _, z1idx = self.Rt.rsample(None, t0, logits=logits1)
            z1 = self.priorz.sample(logits1, (1,)).to(x.device)

            x0 = self.vt.decode(x1, z1)
            
            logits0, z0idx = self.Rt.rsample(x0, t0)
            pi0 = softmax(logits0/self.beta, dim=-1)
            z0 = self.rt.sample(logits0, x0)
            
            # z1 = torch.multinomial(q1, 1).view(-1)
            x0_numpy = x0.cpu().detach().numpy()
            x_numpy = x.cpu().detach().numpy()
            z0_numpy = z0.cpu().detach().numpy()
            pi0_numpy = pi0.cpu().detach().numpy()

            cmap = plt.colormaps['gist_rainbow']

            plt.figure()
            plt.scatter(x0_numpy[:,0], x0_numpy[:,1], c=z0_numpy.squeeze(), marker=".", cmap=cmap)
            plt.colorbar()
            plt.savefig(os.path.join(self.args.log_sample_path, 'image_grid_{}.png'.format(self.current_epoch)))
            plt.close()

            # histogram of the generated class label
            fig, axes = plt.subplots(5, 3, figsize=(5,5), sharex=True, constrained_layout=True)
            for i, rax in enumerate(axes):
                mask = y==i
                xk = x[y==i]
                for j, cax in enumerate(rax):
                    if j<len(xk):
                        pi1 = self.priorpi.sample((100,))
                        x0 = xk[j].unsqueeze(0).to(torch.float64).repeat(100,1)
                        
                        t0 = torch.tensor(0., device=x0.device)
                        logits0, z0idx = self.Rt.rsample(x0, t0)
                        pi0 = softmax(logits0/self.beta, dim=-1)
                        kidx = torch.argmax(pi0, dim=-1).squeeze()
                        cax.hist(kidx, bins=20, alpha=0.7)
                    else:
                        cax.axis("off")
                    if j==0:
                        cax.set_ylabel(f"y={i}")

            plt.tight_layout()
            plt.savefig(os.path.join(self.args.log_sample_path, 'post_grid_{}.png'.format(self.current_epoch)))
            plt.close()

        self.vt.train()
        self.Rt.train()
        self.rt.train()
        # self.priorz.train()

    def test_generate(self, n):
        x1 = torch.randn(n, self.config.data.size)
        # generated samples
        logits1 = self.priorpi.sample((n,)).to(self.device)
        z1idx = F.gumbel_softmax(logits1, tau=self.config.flow.tau, hard=False)
        z1 = self.priorz.sample(z1idx, (1,))

        x0 = self.vt.decode(x1, z1)
        t0 = torch.tensor(0., device=x0.device)

        logits0, z0idx = self.Rt.rsample(x0, t0)
        pi0 = softmax(logits0/self.beta, dim=-1)
        z0 = self.rt.sample(logits0, x0)
        
        return x0, pi0

    def test_step(self, batch, batch_idx):
        X, y = batch  # Assuming the batch is the input data `x`
        self.test_plot_path(X, y)




class PinWheelMixtureRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

        self.k = self.config.data.n_classes
        self.priorz = GaussianMixturePrior(self.config.flow.k_dim, self.config.flow.z_dim, hidden_features=[])

        self.vt = LLK(
            self.config.data.size, 
            self.config.flow.z_dim, 
            model="mlp", 
            hidden_features=[self.config.model.ngf]*4,
            freqs=self.config.model.freqs,
            fct=nn.Tanh(),
            hidden_dim=self.config.data.size,
            )
        
        self.Rt = CatNF_fixed(
            self.config.data.size, 
            self.config.flow.k_dim, 
            fct=nn.Tanh(), 
            # fct=nn.SiLU(),
            freqs=2,
            hidden_features=[self.config.flow.ngf]*0
            )
        self.rt = GaussianMixtureComponent(
            self.config.flow.k_dim, 
            self.config.flow.z_dim, 
            self.config.data.size, 
            fct=nn.Tanh(), 
            # freqs=self.config.flow.freqs,
            freqs=2,
            hidden_dim=self.config.flow.k_dim, # 2,
            hidden_features=[self.config.flow.ngf]*0
            )

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
            strategy=DDPStrategy(find_unused_parameters=True)
            devices="auto"
        else:
            accelerator='cpu'
            devices="auto"
            strategy = "auto"

        llk_callback = PlotLogLikelihoodCallback(save_path=self.args.log_sample_path, log_keys=("tra_log_lik", "tra_log_post_z"))
        # Add the callback

        self.trainer = pl.Trainer(
            max_epochs=self.config.training.n_epochs, 
            # accelerator='gpu',
            accelerator = accelerator,
            devices=devices,
            strategy=strategy,
            callbacks=[llk_callback, self.checkpoint_callback],
        )

    def train(self):
        # load data
        dataset, test_dataset = get_dataset(self.config.data.n_classes, "data", self.config.data.samplesize, self.config.data.test_samplesize)
        train_dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers)
        val_dataloader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=self.config.data.num_workers, drop_last=True)

        # Initialize the Lightning model
        model = PinWheelMixtureLightningModule(self.vt, self.Rt, self.rt, self.priorz, self.config, self.args)
        # Run the training loop
        if not self.args.resume_training:
            ckpt_path = None
        else:
            ckpt_path = self.checkpoint_callback.best_model_path

        self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)

    def sample(self):
        dataset, test_dataset = get_dataset(self.config.data.n_classes, "data", 500, 500)

        test_dataloader = DataLoader(test_dataset, batch_size=2500, shuffle=True,
                        num_workers=self.config.data.num_workers, drop_last=True)

        model = PinWheelMixtureLightningModule(self.vt, self.Rt, self.rt, self.priorz, self.config, self.args)

        ckpt_path = self.checkpoint_callback.best_model_path

        # When loading:
        state_dict = checkpoint['state_dict']
        remapped_state_dict = remap_checkpoint_state_dict(state_dict)
        # Try loading with the remapped state dict
        model.load_state_dict(remapped_state_dict, strict=False)

        self.trainer.test(model, dataloaders=test_dataloader)


class FlowMatchingLightningModule(pl.LightningModule):
    def __init__(self, vt: nn.Module, Rt: nn.Module, rt, priorz, embx, config, args):
        super().__init__()
        self.config = config
        self.args = args
        
        self.vt = vt
        self.Rt = Rt
        self.rt = rt
        self.priorz = priorz
        self.embx = embx
        self.k, self.d = self.config.flow.pi_dim, self.config.flow.z_dim
        self.c, self.p = self.config.data.channel, self.config.data.size
        self.xemb_dim = self.config.model.ngf
        self.sig_min = 1e-4
        # Register buffers for priorz parameters

        self.automatic_optimization = False
        self.last_validation_batch = None
        # temperature annealing parameters
        self.temp_max = self.config.flow.beta
        self.temp_min = 0.5
        self.beta = self.temp_max

    def setup(self, stage=None):
        # Reinitialize the distributions using the buffers now on the correct device
        # self.priorpi = Dirichlet(torch.ones(self.k).to(self.device))
        self.priorpi = Normal(torch.zeros(self.k, device=self.device), torch.ones(self.k, device=self.device))
        if not self.config.model.cnn:
            self.priory = Normal(torch.zeros(self.p**2).to(self.device), torch.ones(self.p**2).to(self.device))
        else:
            self.priory = Normal(torch.zeros(self.c, self.p, self.p).to(self.device), torch.ones(self.c, self.p, self.p).to(self.device))
        
        # anneal beta from large to small
        self.flow_matching_loss = FlowMatchingLossMixture(
                                vt=self.vt, Rt=self.Rt, rt=self.rt, 
                                priorpi=self.priorpi, priorz=self.priorz, priory=self.priory, 
                                k=self.k, alpha=self.config.training.alpha, beta=self.beta)
       

    # def forward(self, pi0, n):

    #     if not self.config.model.cnn:
    #         # for generating data given class z (batched integer)
    #         # y1 = torch.rand(n,self.p**2).to(self.device)
    #         y1 = self.priory.sample((n,)).to(self.device)
    #         x1 = inv_transform(y1)
    #         # pi1 = self.priorpi.sample((n,)).to(self.device)
    #         z1 = self.priorz.sample(pi0, (n,)).to(self.device)

    #         y0 = self.vt.decode(y1, z1)
    #         x0 = inv_transform(y0)

    #         z1_np = z1.cpu().detach().numpy()
    #         x0_np = x0.cpu().detach().numpy().reshape((-1,self.c,self.p,self.p))
    #     else:
    #         # for generating data given class z (batched integer)
    #         y1 = torch.randn(n,self.c,self.p,self.p).to(self.device)
    #         z1 = self.priorz.sample(pi0, (n,)).to(self.device)
    #         # q1 = to_one_hot(z, self.config.data.n_classes)
    #         y0 = self.vt.decode(y1, z1)

    #         # x0 = torch.sigmoid(y0)
    #         x0 = inv_transform(y0)
    #         z1_np = z1.cpu().detach().numpy()
    #         x0_np = x0.cpu().detach().numpy()
    #         # print("x1_output", x1_np.shape)
    #     return x0_np, z1_np
    
    def _get_annealed_values(self):
        progress = self.global_step / max(self.trainer.estimated_stepping_batches, 1)
        temp = max(self.temp_min, self.temp_max * math.exp(-5 * progress))
        return temp
    
    def training_step(self, batch, batch_idx):
        # print("train")
        X, y = batch  # Assuming the batch is the input data `x`
        
        # self.beta = self._get_annealed_values()
        # self.flow_matching_loss.beta = self.beta
        # self.Rt.beta = self.beta

        # print("sample", X[0])
        # print("X", X[0])
        if not self.config.model.cnn:
            X = X.view(-1, self.c*self.p**2)
        else:
            X = X
        loss = self.flow_matching_loss(X)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        opt = self.optimizers()
        opt.zero_grad()

        self.manual_backward(loss)
        opt.step()

        self.clip_gradients(opt, gradient_clip_val=self.config.training.clipval, gradient_clip_algorithm="norm")

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        if not self.config.model.cnn:
            X = X.view(-1, self.c*self.p**2)
        else:
            X = X
        val_loss = self.flow_matching_loss(X)
        # generate posterior and evaluate log_posterior and log_lik
        # Store the last batch for plotting
        if batch_idx == self.trainer.num_val_batches[0] - 1:
            self.last_validation_batch = {"X": X, "y": y}

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(
                self.vt.parameters(),
                self.Rt.parameters(),
                self.rt.parameters(),
                ),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay)
        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = self.config.training.n_epochs,
                            eta_min = 1e-5,
                            last_epoch = -1
                        )

        warmUpScheduler = WarmUpScheduler(
            optimizer = optimizer,
            lr_scheduler=cosineScheduler,
            warmup_steps=self.config.training.n_epochs // 10,
            warmup_start_lr=0.00005,
            len_loader=self.config.data.samplesize//self.config.training.batch_size
            )

        return [optimizer], [
                {'scheduler': warmUpScheduler,
                'monitor': 'train_loss',
                "interval":"epoch",
                "frequency":1}]

    def on_train_epoch_end(self):
        # manual scheduler step
        sch = self.lr_schedulers()
        sch.step()


    def on_validation_epoch_end(self):
        if self.last_validation_batch is not None:
            X = self.last_validation_batch["X"]
            y = self.last_validation_batch["y"]

            if not self.config.model.cnn:
                X = X.view(-1, self.c*self.p**2)

            
            if self.current_epoch % self.config.training.snapshot_freq == 0:
                
                """ Snapshot sampling at the end of every epoch """
                # if self.config.training.snapshot_sampling:
                self.plot_latent(X, y)
                self.generate(X, y)

                 # if self.current_epoch % self.config.training.snapshot_freq == 0:
                z0, logits0, pi0 = self.sample(X)
                # evaluate likelihood
                log_post_z = self.rt.log_prob(logits0, X.flatten(start_dim=1), z0).mean()
                # print("log_post_z", log_post_z)
                log_lik = self.vt.log_prob(X, z0, 0., self.priory).mean()
                self.log('val_log_post_z', log_post_z, on_step=False, on_epoch=True, sync_dist=True, logger=True)
                self.log('val_log_lik', log_lik, on_step=False, on_epoch=True, sync_dist=True, logger=True)

                print("log_post", log_post_z)
                print("log_lik", log_lik)
                
            
        # Clear the stored batch for next epoch
        self.last_validation_batch = None

    def sample(self, x):
        if self.config.flow.cnn:
            _x = x
        else:
            _x = x.flatten(start_dim=1)

        logits1 = self.priorpi.sample((len(x),)).to(self.device)
        pi1 = F.softmax(logits1/self.beta, dim=-1)
        t0 = torch.tensor(0., device=self.device)
        _, z1idx = self.Rt.rsample(None, t0, logits=logits1)
        logits0, z0idx = self.Rt.rsample(_x, t0)
        pi0 = F.softmax(logits0/self.beta, dim=-1)
        z0 = self.rt.sample(logits0, _x).to(self.device)

        return z0, logits0, pi0/pi0.sum(-1, keepdims=True)



    def generate(self, x, y, cmap="gray"):
        # Set models to evaluation mode
        self.vt.eval()
        self.Rt.eval()
        self.rt.eval()
        # self.priorz.eval()
        # self.embx.eval()
        with torch.no_grad():


            fig, axes = plt.subplots(10, 8, figsize=(10, 10))
            for row_idx, row_axes in enumerate(axes):
                m = len(row_axes)
                mask = y==row_idx
                if mask.sum() == 0:
                    pass
                else:
                    
                    x_k = x[mask][0]
                    y_k = to_one_hot(y[y==row_idx], self.k)[0]
                    # x0 = inv_transform(x_k).repeat(m-1,1)
                    if self.config.model.cnn:
                        y0 = x_k.repeat(m-1,1,1,1)
                    else:
                        y0 = x_k.repeat(m-1,1)

                    if self.config.flow.cnn:
                        _y0 = y0
                    else:
                        _y0 = y0.flatten(start_dim=1)
                        

                    logits1 = self.priorpi.sample((m-1,)).to(self.device)
                    t0 = torch.tensor(0., device=self.device)
                    _, z1idx = self.Rt.rsample(None, t0, logits=logits1)
                    logits0, z0idx = self.Rt.rsample(_y0, t0)
                    pi0 = softmax(logits0/self.beta, dim=-1)
                    z0 = self.rt.sample(logits0, _y0)

                    y1_new = self.priory.sample((m-1,)).to(self.device)
                    y0_new = self.vt.decode(y1_new, z0)

                    x0_np = inv_transform(y0_new).cpu().detach().numpy()
                    x_np = inv_transform(x_k).cpu().detach().numpy()

                    if not self.config.model.cnn:
                        x0_np = x0_np.reshape((-1,self.c,self.p,self.p))
                        x_np = x_np.reshape((self.c,self.p,self.p))
                    
                    for col_idx, ax in enumerate(row_axes):   
                        if col_idx == 0:
                            x_np_tr = np.transpose(x_np, (1, 2, 0))
                            ax.imshow(x_np_tr, cmap=cmap)
                            ax.set_ylabel("y={}".format(row_idx))
                        else:
                            x0_np_tr = np.transpose(x0_np[col_idx-1], (1, 2, 0))
                            # print(x0_np_tr)
                            # print()
                            ax.imshow(x0_np_tr, cmap=cmap)
                            ax.set_ylabel("")
                        ax.set_xlabel("")
                        ax.get_xaxis().set_ticks([])
                        ax.get_yaxis().set_ticks([])
            plt.tight_layout()
            # plt.savefig(os.path.join(self.args.log_sample_path, '{}_sampels.png'.format(ckpt_file)))
            plt.savefig(os.path.join(self.args.log_sample_path, f'image_grid_epoch_{self.current_epoch}.png'))
            plt.close()
            

            fig, axes = plt.subplots(10, 3, figsize=(10,10), sharex=True, constrained_layout=True)
            for row_idx, row_axes in enumerate(axes):
                # print(f"Row {row_idx}")
                x_k = x[y==row_idx]
                for col_idx, ax in enumerate(row_axes):   
                    if col_idx < len(x_k):
                        if self.config.model.cnn:              
                            y0 = x_k[col_idx].repeat(100,1,1,1)
                        else:
                            y0 = x_k[col_idx].repeat(100,1)

                        if self.config.flow.cnn:
                            _y0 = y0
                        else:
                            _y0 = y0.flatten(start_dim=1)

                        t0 = torch.tensor(0., device=self.device)
                        logits0, z0idx = self.Rt.rsample(_y0, t0)
                        pi0 = softmax(logits0/self.beta, dim=-1)

                        kidx = torch.argmax(pi0, dim=-1).squeeze()
                        kidx_np = kidx.cpu().detach().numpy()

                        # ax.hist(z0, bins=20, alpha=0.7)
                        ax.hist(kidx_np, bins=20, alpha=0.7)
                    else:
                        ax.axis('off')
                    # ax.set_title(f"y={y[i]}")
                    if col_idx == 0:
                        ax.set_ylabel(f"y={row_idx}")

            # plt.tight_layout(pad=3.0)
            plt.savefig(os.path.join(self.args.log_sample_path, 'postpi_grid_epoch_{}.png'.format(self.current_epoch)))
            plt.close()
        # Optionally, switch back to training mode after sampling
        self.vt.train()
        self.Rt.train()
        self.rt.train()
        # self.priorz.train()
        # self.embx.train()

    def plot_latent(self, x, y):
        # Set models to evaluation mode
        self.vt.eval()
        self.Rt.eval()
        self.rt.eval()
        # self.priorz.eval()

        with torch.no_grad():

            z0, z0idx, pi0 = self.sample(x)

        z0_np = z0.cpu().detach().numpy()

        k0_np = np.argmax(z0idx.cpu().detach().numpy(), axis=1)
        # print(pi0_np)
        y_np = y.cpu().detach().numpy()

        cmap = plt.colormaps['tab10']
        
        if self.config.flow.z_dim == 2:
            z0_proj = z0_np.squeeze()
            plt.figure(figsize=(8, 6))
            plt.scatter(z0_proj[:,0], z0_proj[:,1], c=k0_np, cmap=cmap, s=10)
            # Add the color bar
            plt.colorbar(shrink=0.5, orientation='vertical')
            plt.title("Generated latent z given x")
            plt.tight_layout()
            plt.savefig(os.path.join(self.args.log_sample_path, f'postz_grid_epoch_{self.current_epoch}.png'))
            plt.close()
        elif self.config.flow.z_dim == 3:
            z0_proj = z0_np.squeeze()
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(z0_proj[:,0], z0_proj[:,1], z0_proj[:,2], c=k0_np, cmap=cmap, s=10)
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, orientation='vertical', shrink=0.5)
            ax.set_title("Generated latent z given x")

            plt.tight_layout()
            plt.savefig(os.path.join(self.args.log_sample_path, f'postz_grid_epoch_{self.current_epoch}.png'))
            plt.close()
        else:
            # z0_proj = first_three_eigen_proj(z0_np)
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

        # Optionally, switch back to training mode after sampling
        self.vt.train()
        self.Rt.train()
        self.rt.train()


    def test_plot_latent(self, x, y):
        # sample from posterior
        z0, z0idx, pi0 = self.sample(x)
        z0_np = z0.cpu().detach().numpy()
        k0_np = np.argmax(pi0.cpu().detach().numpy(), axis=1)
        # print(pi0_np)
        y_np = y.cpu().detach().numpy()
        pi_np = pi0.cpu().detach().numpy()

        cmap = plt.colormaps['tab10']
        
        # z0_proj = first_three_eigen_proj(z0_np)
        tsne = TSNE(n_components=2, perplexity=50, random_state=14, n_iter=1000)
        zc_proj = tsne.fit_transform(z0_np)

        plt.figure(figsize=(8,6))
        plt.scatter(zc_proj[:,0], zc_proj[:,1], c=y_np, cmap=cmap, s=10)
        plt.colorbar(shrink=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.log_sample_path, f'eval_grid_epoch_{self.current_epoch}.png'))
        # plt.show()
        plt.close()

        ari = adjusted_rand_score(y_np, k0_np)
        nmi = normalized_mutual_info_score(y_np, k0_np)
        nmi_soft = soft_nmi(pi_np, y_np)
        return nmi, ari, nmi_soft

    def test_generation_figure(self, x, y):
        # assume x cnotains each digit no repeat no missing
        # pick 1 digit each (a) plot original (2) plot one predictive (3) plot one posterior on \xi
        sorted_idx = torch.argsort(y)
        xk = x[sorted_idx]
        if self.config.model.cnn:
            xk = xk.flatten(start_dim=1)
        print("xk", xk.shape)
        row_ratios = [1, 1, 1.5]
        with torch.no_grad():
            fig, axes = plt.subplots(3, self.k, figsize=(10,4), gridspec_kw={'height_ratios': row_ratios}) 

            # generate posterior
            logits1 = self.priorpi.sample((len(xk),)).to(self.device)
            t0 = torch.tensor(0., device=self.device)
            _, z1idx = self.Rt.rsample(None, t0, logits=logits1)
            logits0, z0idx = self.Rt.rsample(xk, t0)
            pi0 = F.softmax(logits0/self.beta, dim=-1)
            z0 = self.rt.sample(logits0, xk)

            y1_new = self.priory.sample((len(xk),)).to(self.device)
            y0_new = self.vt.decode(y1_new, z0)
            xnew = inv_transform(y0_new).reshape((-1,self.c,self.p,self.p))

            xk = inv_transform(xk).reshape((-1,self.c,self.p,self.p))
            # print("xnew", xnew.shape)
            # find probability for each x

            xnew_np = xnew.cpu().detach().numpy()
            xk_np = xk.cpu().detach().numpy()
            prob_np = pi0.cpu().detach().numpy()
            # print("x0", x0.shape)
            
            for col_idx in range(10):
                axes[0, col_idx].imshow(np.transpose(xk_np[col_idx], (1,2,0)), cmap="gray")
                axes[1, col_idx].imshow(np.transpose(xnew_np[col_idx], (1,2,0)), cmap="gray")
                axes[2, col_idx].bar(range(self.k), prob_np[col_idx])
                axes[0, col_idx].get_xaxis().set_ticks([])
                axes[0, col_idx].get_yaxis().set_ticks([])
                axes[1, col_idx].get_xaxis().set_ticks([])
                axes[1, col_idx].get_yaxis().set_ticks([])
                axes[2, col_idx].set_xlim(-1,self.k)
                axes[2, col_idx].set_ylim(0,1)
                if col_idx == 0:
                    axes[0, col_idx].set_ylabel("Real")
                    axes[1, col_idx].set_ylabel("Generated")
                    axes[2, col_idx].set_ylabel(r"$p(\xi|x)$")
                if col_idx != 0:
                    axes[2, col_idx].get_yaxis().set_ticks([])

            plt.tight_layout()
            # plt.savefig(os.path.join(self.args.log_sample_path, '{}_sampels.png'.format(ckpt_file)))
            plt.savefig(os.path.join(self.args.log_sample_path, f'eval_sfa_mixture_eval_display.png'))
            plt.close()


    def test_sample(self, n, x, y):
        # first sample from x
        sorted_idx = torch.argsort(y)
        xk = x[sorted_idx]
        yk = y[sorted_idx].numpy()
        if self.config.model.cnn:
            xk = xk.flatten(start_dim=1)

        fig, axes = plt.subplots(10, n, figsize=(12,16))
        
        # with one sample from each class, see the perturbation
        for i, class_label in enumerate(range(self.k)): # self.config.data.n_classes
            # print(i)
            class_indices = np.where(yk == class_label)[0]
            # print(class_indices)
            top_index = class_indices[-1:]
            
            _xk = xk[top_index].repeat(n,1,1,1)
            # then sample from p(z|x)
            logits1 = self.priorpi.rsample((n,)).to(self.device)
            # z1idx = F.gumbel_softmax(logits1, tau=self.tau, hard=hard)
            t0 = torch.tensor(0., device=self.device)
            _, z1idx = self.Rt.rsample(None, t0, logits=logits1)

            z1 = self.priorz.rsample(z1idx, (1,)).to(self.device)
            logits0, z0idx = self.Rt.rsample(_xk, t0)
            pi0 = softmax(logits0/self.beta, dim=-1)
            z0 = self.rt.sample(logits0, _xk.flatten(start_dim=1))

            x1_new = self.priory.sample((n,)).to(self.device)
            x0_new = self.vt.decode(x1_new, z0)

            _z0 = z0.cpu().detach().numpy()
            _z0idx = z0idx.cpu().detach().numpy()
            _x0_new = inv_transform(x0_new).cpu().detach().numpy()

            for j in range(n):

                axes[i, j].imshow(np.transpose(_x0_new[j], (1,2,0)))
                axes[i, j].axis("off")

        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(self.args.log_sample_path, f'eval_cond_generation.png'))
        plt.close()
        

    def test_step(self, batch, batch_idx):
        X, y = batch
        if not self.config.model.cnn:
            X = X.view(-1, self.config.data.size**2)
        
        if self.args.figure:
            self.test_generation_figure(X, y)
        else:
            """ Snapshot sampling at the end of every epoch """
            # if self.config.training.snapshot_sampling:
            nmi, ard, nmi_soft = self.test_plot_latent(X, y)
            self.log('test_post_clus_nmi', nmi, on_step=True, on_epoch=True, sync_dist=True, logger=True)
            self.log('test_post_clus_ard', ard, on_step=True, on_epoch=True, sync_dist=True, logger=True)
            self.log('test_post_clus_nmi_soft', nmi_soft, on_step=True, on_epoch=True, sync_dist=True, logger=True)




class MNISTMixtureRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

        self.k, self.d = self.config.flow.pi_dim, self.config.flow.z_dim
        self.c, self.p = self.config.data.channel, self.config.data.size
        self.priorz = GaussianMixturePrior(self.k, self.d, hidden_features=[], fct=nn.Softplus())

        vt_type = getattr(self.config.model, 'type', 'cnn')
        if vt_type == 'cnn':
            self.vt = LLK(
                self.p, self.d, model="cnn", in_ch=self.c,
                mod_ch=self.config.model.mod_ch, freqs=self.config.model.freqs,
                hidden_features=[self.config.flow.ngf]*3,
                fct=nn.SiLU(),
                )
        else:
            self.vt = LLK(
                self.p, self.d, model="mlp", is_image=True, in_ch=self.c,
                freqs=self.config.model.freqs,
                hidden_features=[self.config.flow.ngf]*3,
                fct=nn.SiLU(),
                hidden_dim=28*28,
                )
        self.embx = None

        self.rt = GaussianMixtureComponent(
            self.k, self.d, self.c*self.p**2, hidden_features=[self.config.flow.ngf]*0
            , hidden_dim=28*28
            , fct=nn.Tanh()
            , in_ch=self.c
            , freqs=self.config.flow.freqs
            ).to(self.config.device)

        self.Rt = CatNF_fixed(
            self.c*self.p**2, self.k, hidden_features=[self.config.flow.ngf]*0
            , temp=self.config.flow.beta
            , freqs=self.config.flow.freqs
            , fct=nn.Tanh()
            # , fct=nn.Softplus()
            , in_ch=self.c
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
            strategy=DDPStrategy(find_unused_parameters=True)
            devices="auto"
        else:
            accelerator='cpu'
            devices="auto"
            strategy = "auto"

        # Add the callback
        plot_loss_callback = PlotLossCallback(save_path=os.path.join(self.args.log_sample_path, f'loss.png'), update_interval=1)
        plot_llk_callback = PlotLogLikelihoodCallback(save_path=self.args.log_sample_path, log_keys=("val_log_lik", "val_log_post_z"))
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.n_epochs, 
            # accelerator='gpu',
            accelerator = accelerator,
            devices=devices,
            strategy=strategy,
            callbacks=[plot_loss_callback, plot_llk_callback, self.checkpoint_callback],
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
        model = FlowMatchingLightningModule(self.vt, self.Rt, self.rt, self.priorz, self.embx, self.config, self.args)
        # Run the training loop
        if not self.args.resume_training:
            ckpt_path = None
        else:
            ckpt_path = self.checkpoint_callback.best_model_path
        self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


    def sample(self):
        if self.config.data.in_sample:
            dataset, _, sampler, _ = get_mnist(
                    self.config.data.n_classes, "data", 300, 50)
            dataloader = DataLoader(dataset, batch_size=5000,
                    num_workers=self.config.data.num_workers, sampler=sampler)
        else:
            # 0123456789ABCDEFGHIJ
            dataset, sampler = get_emnist(
                20, "data", 300, 50, split="balanced") # balanced
            dataloader = DataLoader(dataset, batch_size=6000,
                                num_workers=self.config.data.num_workers, sampler=sampler)
        ckpt_path = ckpt_path = self.checkpoint_callback.best_model_path

        
        model = FlowMatchingLightningModule(self.vt, self.Rt, self.rt, self.priorz, self.embx, self.config, self.args)

        # set to test mode
        self.trainer.test(model, dataloaders=dataloader, ckpt_path=ckpt_path)

    def draw_figure(self):

        _, test_dataset, _, test_sampler = get_mnist(
                self.config.data.n_classes, "data", 1, 1) # 1,1,10 # 500, 500, 5000
        test_dataloader = DataLoader(test_dataset, batch_size=10,
                num_workers=self.config.data.num_workers, sampler=test_sampler)
        
        ckpt_path = ckpt_path = self.checkpoint_callback.best_model_path

        model = FlowMatchingLightningModule(self.vt, self.Rt, self.rt, self.priorz, self.embx, self.config, self.args)

        self.trainer.test(model, dataloaders=test_dataloader, ckpt_path=ckpt_path)
        
       

