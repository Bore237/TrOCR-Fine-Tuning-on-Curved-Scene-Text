class SCUTDataset(Dataset):
    def __init__(self, root_dir, ann_path, tokenizer, transforms=None):
        self.root_dir = root_dir
        self.ann_path = ann_path
        self.tokenizer = tokenizer
        self.transforms = transforms

        self.samples = self.load_annotations()

    def load_annotations(self):
        # c’est ici qu’on met le code
        pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # on chargera l’image ici
        pass
