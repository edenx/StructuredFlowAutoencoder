import os
import itertools
import numpy as np

import torch
import torch.optim as optim
from utils import *
from Scheduler import WarmUpScheduler
from sfa_llk import LLK
from sfa_lds import *
from callbacks import PlotLogLikelihoodCallback, PlotLossCallback, PlotRMSECallback, GradientNormPlotCallback

from dataloader.dataloader_pendulum import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import gc
gc.collect()

torch.set_printoptions(precision=3)
torch.set_default_dtype(torch.float64)

# Define your LightningModule
class FlowMatchingLightningModule(pl.LightningModule):
    def __init__(self, vt: nn.Module, rt: nn.Module, prior, config, args):
        super().__init__()
        self.config = config
        self.args = args
        self.lr = self.config.optim.lr
        self.prior = prior
        self.vt = vt
        self.rt = rt
        self.c, self.p = self.config.data.channel, self.config.data.p
        self.S = self.config.data.S

        self.automatic_optimization = False
        self.last_validation_batch = None

        self.q = self.config.data.q

    def setup(self, stage=None):
        if not self.config.model.cnn:
            self.priory = Normal(torch.zeros(self.p**2).to(self.device), torch.ones(self.p**2).to(self.device))
        else:
            self.priory = Normal(torch.zeros(self.c, self.p, self.p).to(self.device), torch.ones(self.c, self.p, self.p).to(self.device))
        
        self.flow_matching_loss = FlowMatchingLossSeq(
            self.vt, self.rt, self.prior,
            alpha=self.config.training.alpha,
            const=self.config.model.const,
            flowcnn=self.config.flow.cnn)


    def forward(self, n, x, z, indices=None):
        # for generating data given class z (batched integer)
        z0 = self.prior.sample(n, self.config.data.S, device=self.device)

        if not self.config.flow.cnn:
            x0 = torch.randn(self.config.data.S, n, 1, self.config.data.p, self.config.data.p, device=self.device) # channel=1
            x1 = self.vt.decode_sequence(x0, z0)
            z1 = self.rt.decode_sequence(z0, x1.flatten(start_dim=2), indices=indices)
        else:  
            x0 = torch.randn(self.config.data.S, n, 1, self.config.data.p, self.config.data.p, device=self.device) # channel=1
            x1 = self.vt.decode_sequence(x0, z0)
            z1 = self.rt.decode_sequence(z0, x1, indices=indices)

        x1_np = inv_transform(x1).cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.data.p, self.config.data.p))
        z1_np = z1.cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.flow.feature_dim))
        z0_np = z0.cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.flow.feature_dim))
        
        return x1_np, z1_np, z0_np

    def training_step(self, batch, batch_idx):
        # print("train")
        X, y, indices = batch  # Assuming the batch is the input data `x`
        X = X.view(self.config.data.S, -1, 1, self.config.data.p, self.config.data.p).to(dtype=torch.float64)
        indices = indices.view(self.config.data.S, -1)
        
        loss = self.flow_matching_loss(X, indices=indices)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=self.config.training.clipval, gradient_clip_algorithm="norm")
        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        X, y, indices = batch

        X = X.view(self.config.data.S, -1, 1, self.config.data.p, self.config.data.p).to(dtype=torch.float64)
        indices = indices.view(self.config.data.S, -1)
        val_loss = self.flow_matching_loss(X, indices=indices)

        # Store the last batch for plotting
        if batch_idx == self.trainer.num_val_batches[0] - 1:
            self.last_validation_batch = {"X": X, "y": y, "indices":indices}
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(
                self.vt.parameters(),
                self.rt.parameters(),
                ),
            lr=self.lr,
            weight_decay=self.config.optim.weight_decay)
        cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = self.config.training.n_epochs,
                            eta_min = 1e-6,
                            last_epoch = -1
                        )
        warmUpScheduler = WarmUpScheduler(
            optimizer = optimizer,
            lr_scheduler=cosineScheduler,
            warmup_steps=5,
            warmup_start_lr=0.00001,
            len_loader=1  # step() is called once per epoch from on_train_epoch_end
            )

        return [optimizer], [
                {'scheduler': warmUpScheduler,
                'monitor': 'train_loss',
                "interval":"epoch",
                "frequency":1}]



    def eval_latent(self, x, y, indices):
        # Set models to evaluation mode
        self.vt.eval()
        self.rt.eval()
        # self.prior.eval()
        n = x.shape[1]
        with torch.no_grad():
            # find the latent z given data x
            z0 = self.prior.sample(n, self.config.data.S, device=self.device) # S,n,p
            x0 = torch.randn(self.config.data.S, n, 1, self.config.data.p, self.config.data.p, device=self.device) # channel=1

            if self.config.flow.cnn:
                z1 = self.rt.decode_sequence(z0, x, indices=indices)
            else:
                z1 = self.rt.decode_sequence(z0, x.flatten(start_dim=2), indices=indices)

            if self.config.model.const:
                x1 = self.vt.decode_sequence(x0, z1)
            else:
                x1 = self.vt.decode_sequence(x0, z1, indices=indices)

            x1_np = inv_transform(x1).cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.data.p,self.config.data.p))
            z1_np = z1.cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.flow.feature_dim))
            y_np = y.cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.data.n))
            x_np = x.cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.data.p,self.config.data.p))

            plot_image_sequence_and_trajectory(x1_np[0], z1_np[0], figsize=(20,2))
            # # plt.savefig(os.path.join(self.args.log_sample_path, '{}_sampels.png'.format(ckpt_file)))
            plt.savefig(os.path.join(self.args.log_sample_path, f'image_eval_epoch_{self.current_epoch}.png'))
            plt.close()

            plot_image_sequence_and_trajectory(x_np[0], y_np[0], figsize=(20,2))
            # plt.savefig(os.path.join(self.args.log_sample_path, '{}_sampels.png'.format(ckpt_file)))
            plt.savefig(os.path.join(self.args.log_sample_path, f'image_true.png'))
            plt.close()
        
        # evaluate L2 distance between generated and truth
        rmse_obs = np.sqrt((((x1_np-x_np)**2).sum((-1))).mean(1)).mean()
        rmse_latent = np.sqrt((((z1_np[:,:,:2]-y_np)**2).sum((-1))).mean(1)).mean()

        # Optionally, switch back to training mode after sampling
        self.vt.train()
        self.rt.train()

        return rmse_obs, rmse_latent


    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()


    def on_validation_epoch_end(self):
        if self.last_validation_batch is not None:
            X = self.last_validation_batch["X"]
            y = self.last_validation_batch["y"]
            indices = self.last_validation_batch["indices"]
            
            """ Snapshot sampling at the end of every epoch """
            if self.current_epoch % self.config.training.snapshot_freq == 0:
                rmse_obs, rmse_latent = self.eval_latent(X, y, indices)
            
                self.log('rmse_obs', rmse_obs, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
                self.log('rmse_latent', rmse_latent, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

        self.last_validation_batch = None


    def test_step(self, batch, batch_idx):
        x, y, indices = batch  # Assuming the batch is the input data `x`
        F = self.config.data.Stotal
        n = x.shape[0]

        if self.args.sample:
            x = x.view(self.config.data.S, -1, 1, self.config.data.p, self.config.data.p).to(dtype=torch.float64)
            indices = indices.view(self.config.data.S, -1)

            
            z0 = self.prior.sample(n, self.config.data.S, device=self.device) # S,n,p
            x0 = torch.randn(self.config.data.S, n, 1, self.config.data.p, self.config.data.p, device=self.device) # channel=1
            
            if self.config.flow.cnn:
                z1 = self.rt.decode_sequence(z0, x, indices=indices)
            else:
                z1 = self.rt.decode_sequence(z0, x.flatten(start_dim=2), indices=indices)

            if self.config.model.const:
                x1 = self.vt.decode_sequence(x0, z1)
            else:
                x1 = self.vt.decode_sequence(x0, z1, indices=indices)

            x1_np = inv_transform(x1).cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.data.p,self.config.data.p))
            z1_np = z1.cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.flow.feature_dim))
            y_np = y.cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.data.n))
            x_np = x.cpu().detach().numpy().reshape((-1,self.config.data.S,self.config.data.p,self.config.data.p))

            # evaluate L2 distance between generated and truth
            rmse_obs = np.sqrt((((x1_np-x_np)**2).sum((-1))).mean(1)).mean()
            rmse_latent = np.sqrt((((z1_np[:,:,:2]-y_np)**2).sum((-1))).mean(1)).mean()

            self.log('rmse_latent', rmse_latent, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
            self.log('rmse_obs', rmse_obs, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

            self.eval_latent(x, y, indices)
        elif self.args.predict:
            q = np.floor(np.sqrt(F)).astype(np.int32)
            # forward prediction, take the first length xS subsequence as context, then forward prediction
            x = x.view(F, n, 1, self.config.data.p, self.config.data.p).to(dtype=torch.float64)

            z0 = self.prior.sample(n, F, device=self.device) # S,n,p

            x_ = x[:self.config.data.S] # .flatten(start_dim=2)
            x0 = torch.randn(F, n, 1, self.config.data.p, self.config.data.p, device=self.device) # channel=1

            if self.config.flow.cnn:
                z1 = self.rt.predict_future(F, z0, x_)
            else:
                z1 = self.rt.predict_future(F, z0, x_.flatten(start_dim=2))

            # posterior forward prediction
            x1 = self.vt.decode_sequence(x0, z1)

            x1_np = inv_transform(x1).cpu().detach().numpy().reshape((-1,F,self.config.data.p,self.config.data.p))
            z1_np = z1.cpu().detach().numpy().reshape((-1,F,self.config.flow.feature_dim))
            y_np = y.cpu().detach().numpy().reshape((-1,F,self.config.data.n))
            x_np = x.cpu().detach().numpy().reshape((-1,F,self.config.data.p,self.config.data.p))

            x1_stack = x1_np[1][:q**2].reshape(q, self.config.data.p*q, self.config.data.p) # first 7**2=49 frames
            x1_imgrid = x1_stack.swapaxes(0, 1).reshape(self.config.data.p * q, self.config.data.p * q)

            x_stack = x_np[1][:q**2].reshape(q, self.config.data.p*q, self.config.data.p) # first 49 frames
            x_imgrid = x_stack.swapaxes(0, 1).reshape(self.config.data.p * q, self.config.data.p * q)

            fig, axes = plt.subplots(2,2, figsize=(12, 5))
            # plt.hist2d(*x.T, bins=64)
            axes[0,0].imshow(x_imgrid, cmap="gray", origin="lower", aspect=.2)
            axes[0,1].imshow(x1_imgrid, cmap="gray", origin="lower", aspect=.2)
            axes[1,0].plot(np.arange(F), y_np[1])
            # axes[2].plot(y1_np[0][:self.q**2,0], y1_np[0][:self.q**2,1])
            axes[1,0].set_xmargin(0)
            axes[1,1].plot(np.arange(F), z1_np[1])
            # axes[1].plot(z1_np[0][:self.q**2,0], z1_np[0][:self.q**2,1])
            axes[1,1].set_xmargin(0)
            
            axes[0,0].set_xlabel("real data")
            axes[0,1].set_xlabel("gen data")
            axes[1,0].set_xlabel("real latent")
            axes[1,1].set_xlabel("gen latent")
            plt.tight_layout()

            # plot_image_sequence_and_trajectory(x1_np[0], z1_np[0], figsize=(20,2))
            # # # plt.savefig(os.path.join(self.args.log_sample_path, '{}_sampels.png'.format(ckpt_file)))
            plt.savefig(os.path.join(self.args.log_sample_path, f'predict_{F}.png'))
            plt.close()
        



class PENDULUMRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)
        # os.makedirs(self.args.tb_path)

        self.S = self.config.data.S
        self.F = self.config.data.Stotal
        
        # Initialize models `vt` and `rt`
        self.vt = LLK(
            self.config.data.p, self.config.flow.feature_dim,
            model=self.config.model.type, S=self.S, F=self.F,
            is_image=True,
            hidden_dim=100,
            hidden_features=[self.config.model.ngf]*self.config.model.depth,
            fct=nn.SiLU(),
            freqs=self.config.model.freqs,
            dsemb=self.config.model.dsemb
            )
        
        if self.config.flow.cnn:
            self.rt = fullGauss(
                self.config.data.p, self.config.flow.feature_dim, self.S, self.F, dsemb=self.config.flow.dsemb, 
                num_hidden=64,
                num_layers=1,
                freqs=self.config.flow.freqs,
                hidden_features=[self.config.flow.ngf]*self.config.flow.depth,
                fct=nn.Tanh(),
                )
        else:
            print("using fullGauss")
            self.rt = fullGauss(
                self.config.data.p**2, self.config.flow.feature_dim, self.S, self.F, dsemb=self.config.flow.dsemb, 
                num_hidden=64, #64#32,#16,# 14,#16,#32, # 16, # 8,
                num_layers=2, # 3,
                freqs=self.config.flow.freqs,
                cnn=False,
                attention=True,
                hidden_features=[self.config.flow.ngf]*self.config.flow.depth,
                fct=nn.Tanh(),
                )

        
        self.prior = LatentDynamicalSystem(self.config.flow.feature_dim)
        
        # self.dataset, self.val_dataset = get_pendulum(self.config.data.samplesize, self.config.data.p, self.S, "data", gen=self.config.data.gen, plot=False)
        # Define the ModelCheckpoint callback
        self.checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',  # Metric to monitor
            dirpath=self.args.log_path,  # Directory where checkpoints will be saved
            filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',  # Filename convention
            save_top_k=1,  # Only save the best model based on val_loss
            mode='min'  # Minimize the validation loss
        )

        if torch.cuda.is_available():
            accelerator='gpu'
            strategy="ddp"
            devices="auto"
        else:
            accelerator='cpu'
            devices="auto"
            strategy = "auto"

        plot_loss_callback = PlotLossCallback(save_path=os.path.join(self.args.log_sample_path, f'loss.png'), update_interval=1, logy=True)
        plot_llk_callback = PlotLogLikelihoodCallback(save_path=self.args.log_sample_path, log_keys=("val_log_lik", "val_log_post"))
        plot_gradnorm_callback = GradientNormPlotCallback(save_path=os.path.join(self.args.log_sample_path, f'gradnorm.png'))
        plot_rmse_callback = PlotRMSECallback(save_path=os.path.join(self.args.log_sample_path, f'rmse.png'), snapshot_freq=self.config.training.snapshot_freq)
        self.trainer = pl.Trainer(
            # gradient_clip_val=self.config.optim.clip_value,
            max_epochs=self.config.training.n_epochs,
            # accelerator='gpu',
            accelerator = accelerator,
            devices=devices,
            strategy=strategy,
            callbacks=[plot_loss_callback, plot_gradnorm_callback, plot_rmse_callback, self.checkpoint_callback],
        )

        
    def train(self):
        self.length = self.config.data.Stotal # 100 # self.S + 1*(self.config.training.batch_size-1)
        train_dataloader, val_dataloader = get_pendulum_dataloader(self.config.data.samplesize, self.config.data.test_samplesize, 
            self.config.data.p, self.length, "data", 
            window_size=self.S, stride=self.config.data.stride, batch_size=self.config.training.batch_size,
            batch_from_same_trajectory=self.config.data.samebatch,
            state_dim=self.config.data.n, gen=self.config.data.gen, shuffle=True, num_workers=self.config.data.num_workers)

        # Initialize the Lightning model
        model = FlowMatchingLightningModule(self.vt, self.rt, self.prior, self.config, self.args)
        # Run the training loop
        if not self.args.resume_training:
            ckpt_path = None
        else:
            ckpt_path = self.checkpoint_callback.best_model_path

        self.trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)


        # Optionally, run the test loop if a test set is provided
        # trainer.test(model, test_dataloader)

    def sample(self):
        self.length = self.config.data.Stotal 
        train_dataloader, val_dataloader = get_pendulum_dataloader(self.config.data.samplesize, self.config.data.test_samplesize, 
            self.config.data.p, self.length, "data", 
            window_size=self.S, stride=1, batch_size=self.config.training.batch_size,
            batch_from_same_trajectory=self.config.data.samebatch,
            state_dim=self.config.data.n, gen=self.config.data.gen, shuffle=True, num_workers=self.config.data.num_workers)


        # Initialize the Lightning model
        model = FlowMatchingLightningModule(self.vt, self.rt, self.prior, self.config, self.args)
       
        ckpt_path = self.checkpoint_callback.best_model_path

        self.trainer.test(model, val_dataloader, ckpt_path=ckpt_path)

    def predict(self):
        self.length = self.config.data.Stotal 
        _, test_dataloader = get_pendulum_dataloader(1, 2, 
            self.config.data.p, self.length, "data", 
            window_size=self.F, stride=1, batch_size=self.config.training.batch_size,
            batch_from_same_trajectory=self.config.data.samebatch,
            state_dim=self.config.data.n, gen=self.config.data.gen, shuffle=True, num_workers=self.config.data.num_workers)

        # Initialize the Lightning model
        model = FlowMatchingLightningModule(self.vt, self.rt, self.prior, self.config, self.args)
        
        ckpt_path = self.checkpoint_callback.best_model_path

        self.trainer.test(model, test_dataloader, ckpt_path=ckpt_path)

