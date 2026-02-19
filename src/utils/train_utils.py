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
