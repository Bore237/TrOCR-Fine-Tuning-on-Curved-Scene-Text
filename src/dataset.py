import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class SCUTDataset(Dataset):
    def __init__(self, root_dir, ann_path, processor,  train_transforms, max_target_length = 128):
        self.root_dir = root_dir
        self.ann_path = ann_path
        self.processor = processor

        self.samples = self.load_annotations()
        self.max_target_length = max_target_length
        self.transform = train_transforms

    def load_annotations(self):
        samples = []

        with open(self.ann_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # parcourir les lignes
        for line in lines:
            image_name, text = line.strip().split("\t")
            image_path = self.root_dir + image_name

            sample = {
                "image_path": image_path,
                "text": text
            }

            samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        pixel_values =  self.processor(img, return_tensors='pt').pixel_values

        labels = self.processor.tokenizer(
            sample["text"],
            padding='max_length',
            max_length=self.max_target_length
        ).input_ids

        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }

