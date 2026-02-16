import torch
from tqdm import tqdm

def train_one_epoch(model, train_dataloader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc="Training"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(device):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_dataloader)