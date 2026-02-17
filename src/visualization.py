import os
import matplotlib.pyplot as plt


class MetricLogger:
    def __init__(self, save_dir="plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            "train_loss": [],
            "val_wer": [],
            "val_cer": []
        }

    def log(self, train_loss=None, val_wer=None, val_cer=None):
        if train_loss is not None:
            self.history["train_loss"].append(train_loss)
        if val_wer is not None:
            self.history["val_wer"].append(val_wer)
        if val_cer is not None:
            self.history["val_cer"].append(val_cer)

    def plot(self, metrics_to_plot=None):
        if metrics_to_plot is None:
            metrics_to_plot = ["train_loss", "val_wer", "val_cer"]

        for metric in metrics_to_plot:
            values = self.history.get(metric, None)
            if values is None or len(values) == 0:
                continue

            plt.figure(figsize=(8, 5))
            plt.plot(values, marker="o")
            plt.title(metric.replace("_", " ").upper())
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.grid(True)

            save_path = os.path.join(self.save_dir, f"{metric}.png")
            plt.savefig(save_path)
            plt.close()


"""
from visualization import MetricLogger

logger = MetricLogger(save_dir=config["visualization"]["plot_dir"])

# dans la boucle d'entraînement :
logger.log(train_loss=train_loss, val_wer=val_metrics["wer"], val_cer=val_metrics["cer"])

# après la boucle :
logger.plot(metrics_to_plot=config["visualization"]["metrics_to_plot"])

"""