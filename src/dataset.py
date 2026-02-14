import os
from PIL import Image


class SCUTDataset(Dataset):
    def __init__(self, root_dir, ann_path, tokenizer, transforms=None):
        self.root_dir = root_dir
        self.ann_path = ann_path
        self.tokenizer = tokenizer
        self.transforms = transforms

        self.samples = self.load_annotations()

    def load_annotations(self):
        samples = []

        with open(self.ann_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # parcourir les lignes
        for line in lines:
            image_name, text = line.strip().split("\t")
            image_path = os.path.join(self.root_dir, image_name)

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

        if self.transforms:
            pixel_values = self.transforms(img)
        else:
            pixel_values = img

        labels = self.tokenizer(
            sample["text"],
            return_tensors="pt"
        ).input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

