import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class PlotLogLikelihoodCallback(Callback):
    def __init__(self, save_path="loss_plot.png", log_keys=("log_likelihood_x", "log_likelihood_z")):
        super().__init__()
        self.save_path = os.path.join(save_path, f'llk.png')
        self.log_keys = log_keys
        self.log_likelihoods_x = []
        self.log_likelihoods_z = []

    def on_train_epoch_end(self, trainer, pl_module):
        if self.log_keys[0] in trainer.callback_metrics:
            log_likelihood_x = trainer.callback_metrics[self.log_keys[0]].item()
            self.log_likelihoods_x.append(log_likelihood_x)

        if self.log_keys[1] in trainer.callback_metrics:
            log_likelihood_z = trainer.callback_metrics[self.log_keys[1]].item()
            self.log_likelihoods_z.append(log_likelihood_z)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.log_likelihoods_x, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Log p(x|z)")
        plt.title("Log Likelihood of x During Training")
        plt.grid()

        plt.subplot(1, 2, 2)
        plt.plot(self.log_likelihoods_z, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Log p(z|x)")
        plt.title("Log Likelihood of z During Training")
        plt.grid()

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()


class PlotLossCallback(Callback):
    def __init__(self, save_path="loss_plot.png", update_interval=1, logy=False):
        super().__init__()
        self.save_path = save_path
        self.update_interval = update_interval
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.logy = logy

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_num = trainer.current_epoch
        metrics = trainer.callback_metrics

        train_loss = metrics.get("train_loss")
        if train_loss is not None:
            if len(self.train_losses) <= epoch_num:
                self.train_losses.append(train_loss.item())

        val_loss = metrics.get("val_loss")
        if val_loss is not None:
            if len(self.val_losses) <= epoch_num:
                self.val_losses.append(val_loss.item())

        if len(self.epochs) <= epoch_num:
            self.epochs.append(epoch_num)

        if epoch_num % self.update_interval == 0:
            self.plot_and_save()

    def plot_and_save(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label="Training Loss", marker="o")
        plt.plot(self.epochs, self.val_losses, label="Validation Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        if self.logy:
            plt.yscale('log')
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(self.save_path)
        plt.close()


class PlotRMSECallback(Callback):
    def __init__(self, save_path="rmse_plot.png", snapshot_freq=1):
        super().__init__()
        self.save_path = save_path
        self.snapshot_freq = snapshot_freq
        self.rmse_obs_list = []
        self.rmse_latent_list = []
        self.epochs = []

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch_num = trainer.current_epoch
        if epoch_num % self.snapshot_freq != 0:
            return
        metrics = trainer.callback_metrics
        rmse_obs = metrics.get("rmse_obs")
        rmse_latent = metrics.get("rmse_latent")
        if rmse_obs is not None and rmse_latent is not None:
            self.epochs.append(epoch_num)
            self.rmse_obs_list.append(rmse_obs.item())
            self.rmse_latent_list.append(rmse_latent.item())
            self.plot_and_save()

    def plot_and_save(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(self.epochs, self.rmse_obs_list, marker="o")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("RMSE")
        axes[0].set_title("Observation RMSE (generated vs real)")
        axes[0].grid(True)

        axes[1].plot(self.epochs, self.rmse_latent_list, marker="o")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("RMSE")
        axes[1].set_title("Latent RMSE (inferred vs true)")
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close()


class GradientNormPlotCallback(pl.Callback):
    def __init__(self, save_path="gradnorm_plot.png"):
        super().__init__()
        self.save_path = save_path
        self.epoch_grad_norms = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        total_norm = 0.0
        count = 0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()
                count += 1

        avg_grad_norm = total_norm / count if count > 0 else 0.0

        if not hasattr(pl_module, 'batch_grad_norms'):
            pl_module.batch_grad_norms = []
        pl_module.batch_grad_norms.append(avg_grad_norm)

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(pl_module, 'batch_grad_norms') and pl_module.batch_grad_norms:
            epoch_mean = sum(pl_module.batch_grad_norms) / len(pl_module.batch_grad_norms)
        else:
            epoch_mean = 0.0

        self.epoch_grad_norms.append(epoch_mean)
        pl_module.batch_grad_norms = []

        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(self.epoch_grad_norms) + 1), self.epoch_grad_norms, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Mean Gradient Norm")
        plt.title("Mean Gradient Norm per Epoch")
        plt.grid(True)
        plt.savefig(self.save_path)
        plt.close()
