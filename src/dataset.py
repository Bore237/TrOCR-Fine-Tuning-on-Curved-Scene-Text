from PIL import Image
from torch.utils.data import Dataset
import torch

class SCUTDataset(Dataset):
    def __init__(self, root_dir, ann_path, processor, train_transforms=None, max_target_length=160, data_collator = True):
        self.root_dir = root_dir
        self.ann_path = ann_path
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = train_transforms
        self.samples = self.load_annotations()
        self.data_collator =  data_collator

    def load_annotations(self):
        samples = []
        with open(self.ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2: continue  # Skip malformed lines
                
                image_name, text = parts[0], parts[1]
                # Use os.path.join for better path handling
                import os
                image_path = os.path.join(self.root_dir, image_name)
                samples.append({"image_path": image_path, "text": text})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        
        # Apply transforms (Augmentations)
        if self.transform:
            img = self.transform(img)

        # Correct way to get pixel_values for TrOCR
        pixel_values = self.processor(img, return_tensors="pt").pixel_values.squeeze()

        # Tokenize text
        labels = self.processor.tokenizer(
            sample["text"],
            add_special_tokens=True,  # Add BOS and EOS (start and end Token)
            max_length=self.max_target_length,
            truncation=True  # Added truncation to be safe
            #padding= 'max_length', # Dynamic padding (pad to longest in batch) -> DataCollator
        
        ).input_ids

        # Replace padding token id with -100 when we doon't use DataCollator
        if not self.data_collator:
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
            labels = torch.tensor(labels)
            
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }