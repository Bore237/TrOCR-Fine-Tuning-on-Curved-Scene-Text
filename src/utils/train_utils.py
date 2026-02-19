import torch
from tqdm import tqdm
from torch.amp.autocast_mode import autocast

def compute_ocr_metric(processor, pred_ids, label_ids, metrics):
    # Load metric
    wer_metric, cer_metric = metrics

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    # Clone pour éviter modification en place
    label_ids = label_ids.clone()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return cer, wer

def train_scale_one_epoch(model, processor, train_dataloader, optimizer, scheduler, device, scaler, metrics, config):
    model.train()
    
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
            cer, wer = compute_ocr_metric(processor, pred_ids, labels, metrics)
            all_wers += wer
            all_cers += cer

    return {
            "loss": total_loss / len(train_dataloader),
            "cer": all_cers / len(train_dataloader),
            "wer": all_wers / len(train_dataloader)
    }

def evaluate_one_epoch(model, processor, val_loader, device, metrics):
    model.eval()
    all_wers = 0.0
    all_cers = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            label_ids = batch["labels"].to(device)

            # Génération
            pred_ids = model.generate(pixel_values)

            # Calcul CER/WER
            cer, wer = compute_ocr_metric(processor, pred_ids, label_ids, metrics)
            all_wers += wer
            all_cers += cer

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
