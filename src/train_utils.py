import torch
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
from transformers import TrainerCallback
import evaluate

def wapper_compute_metrics(processor):
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        # Force predictions to CPU and convert to NumPy
        # This usually solves the OverflowError
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
    
        # Clean up predictions and labels
        predictions[predictions == -100] = processor.tokenizer.pad_token_id 
        labels[labels == -100] = processor.tokenizer.pad_token_id
    
        # Decode
        pred_str = processor.batch_decode(predictions, skip_special_tokens=True)
        label_str = processor.batch_decode(labels, skip_special_tokens=True) # ignorere les carractère speciaux et autre
    
        # Calculate CER
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
        return {"cer": cer, "wer": wer}
    return compute_metrics

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    #labels = [torch.tensor(item["labels"]) for item in batch]
    labels = [item["labels"].clone().detach()  if isinstance(item["labels"], torch.Tensor) 
                else torch.tensor(item["labels"])  for item in batch]
    
    # Dynamic padding dans fill pad to -100 to avoid training by loss
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100
    )
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

#------------------------------------
#   CallBack
#----------------------------------
class LrLoggerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        optimizer = kwargs['optimizer']
        lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"Epoch {int(state.epoch)} - Learning rates: {lrs}")


class UnfreezeCallback(TrainerCallback):
    def __init__(self, unfreeze_epoch=1, lastfreeze_epoch = 30):
        self.unfreeze_epoch = unfreeze_epoch
        self.lastfreeze_epoch = lastfreeze_epoch
        self.applied = False
        self.applied_all = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if int(state.epoch) == self.unfreeze_epoch and not self.applied:
            model = kwargs['model']
            print(f"Epoch {self.unfreeze_epoch}: unfreezing last half of the encoder")

            # Récupérer toutes les couches de l'encoder
            encoder_layers = list(model.encoder.encoder.layer)
            num_layers = len(encoder_layers)
            num_to_unfreeze = num_layers // 2  # dégel de la moitié des couches
            
            # On gèle la première moitié des couches
            for layer in encoder_layers[:num_layers - num_to_unfreeze]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            # On garde la dernière moitié des couches entraînables
            for layer in encoder_layers[num_layers - num_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True

            self.applied = True

        if int(state.epoch) == self.lastfreeze_epoch and not self.applied_all:
            print(f"Epoch {self.unfreeze_epoch}: unfreezing all the encoder")
            
            model = kwargs['model']
            print(f"Epoch {self.unfreeze_epoch}: unfreezing encoder")
            for param in model.encoder.parameters():
                param.requires_grad = True
            self.applied_all = True

#-----------------------------------------------------------------------
#---------------------- 
#   Manual Loops
#--------------------
def train_scale_one_epoch(model, processor, train_dataloader, optimizer, scheduler, device, scaler, metrics, config):
    model.train()
    compute_metrics = wapper_compute_metrics(processor)
    
    torch.set_grad_enabled(True)
    total_loss = 0
    all_wers = 0.0
    all_cers = 0.0

    for batch in tqdm(train_dataloader, desc="Training"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Forward en mixed precision
        with autocast(device, enabled=config["training"]["mixed_precision"]):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

        with torch.no_grad():
            pred_ids = model.generate(pixel_values=pixel_values)
            ocr_metric = compute_metrics((pred_ids, labels))
            all_wers += ocr_metric["wer"]
            all_cers += ocr_metric['cer']

    return {
            "loss": total_loss / len(train_dataloader),
            "cer": all_cers / len(train_dataloader),
            "wer": all_wers / len(train_dataloader)
    }

def evaluate_one_epoch(model, processor, val_loader, device, metrics):
    model.eval()
    all_wers = 0.0
    all_cers = 0.0
    compute_metrics = wapper_compute_metrics(processor)

    
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            label_ids = batch["labels"].to(device)

            # Génération
            pred_ids = model.generate(pixel_values)

            # Calcul CER/WER
            ocr_metric = compute_metrics((pred_ids, labels))
            all_wers += ocr_metric["wer"]
            all_cers += ocr_metric['cer']

    return {"wer": all_wers / len(val_loader), "cer": all_cers / len(val_loader) }

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min", save_path=None):
        """
        Args:
            patience (int): epochs to wait before stopping
            min_delta (float): minimum improvement required
            mode (str): "min" or "max"
            save_path (str): path to save best model
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric, model,  epoch, optimizer):

        if self.best_score is None:
            self.best_score = metric
            self._save_model(model, epoch, optimizer)
            return

        improvement = (
            metric < self.best_score - self.min_delta
            if self.mode == "min"
            else metric > self.best_score + self.min_delta
        )

        if improvement:
            self.best_score = metric
            self.counter = 0
            self._save_model(model, epoch, optimizer)
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

    def _save_model(self, model, epoch, optimizer):
        if self.save_path and model is not None:
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, self.save_path)
