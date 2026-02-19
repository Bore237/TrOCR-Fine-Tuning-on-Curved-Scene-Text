import os
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, save_dir="plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            "train_loss": [],
            "train_wer": [],
            "val_wer": [],
            "train_cer": [],
            "val_cer": []
        }

    def log(self, train_loss=None, val_wer=None, val_cer=None,
            train_wer=None, train_cer=None):

        if train_loss is not None:
            self.history["train_loss"].append(train_loss)
        if train_wer is not None:
            self.history["train_wer"].append(train_wer)
        if val_wer is not None:
            self.history["val_wer"].append(val_wer)
        if train_cer is not None:
            self.history["train_cer"].append(train_cer)
        if val_cer is not None:
            self.history["val_cer"].append(val_cer)


    def plot(self):
        # Plot Train Loss
        if len(self.history["train_loss"]) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(self.history["train_loss"], marker="o", label="Train Loss")
            plt.title("TRAIN LOSS")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.save_dir, "train_loss.png"))
            plt.close()

        # Plot WER (Train + Val)
        if len(self.history["train_wer"]) > 0 and len(self.history["val_wer"]) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(self.history["train_wer"], marker="o", label="Train WER")
            plt.plot(self.history["val_wer"], marker="o", label="Val WER")
            plt.title("WER")
            plt.xlabel("Epoch")
            plt.ylabel("WER")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.save_dir, "wer.png"))
            plt.close()

        # Plot CER (Train + Val)
        if len(self.history["train_cer"]) > 0 and len(self.history["val_cer"]) > 0:
            plt.figure(figsize=(8, 5))
            plt.plot(self.history["train_cer"], marker="o", label="Train CER")
            plt.plot(self.history["val_cer"], marker="o", label="Val CER")
            plt.title("CER")
            plt.xlabel("Epoch")
            plt.ylabel("CER")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(self.save_dir, "cer.png"))
            plt.close()

"""
from visualization import MetricLogger

logger = MetricLogger(save_dir=config["visualization"]["plot_dir"])

# dans la boucle d'entraînement :
logger.log(train_loss=train_loss, val_wer=val_metrics["wer"], val_cer=val_metrics["cer"])

# après la boucle :
logger.plot(metrics_to_plot=config["visualization"]["metrics_to_plot"])

"""