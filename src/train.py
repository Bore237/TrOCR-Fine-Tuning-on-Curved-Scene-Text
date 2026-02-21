from torch.utils.data import DataLoader
from src import SCUTDataset
import yaml
from src import load_model
from torchvision import transforms

def main(config_path):

      # -----------------------------
    # Load config path
    # -----------------------------
    with open(config_path, "r") as f: 
        config = yaml.safe_load(f)
    # -----------------------------
    # Load model + processor
    # -----------------------------
    processor, model, device = load_model(model_name = config["model"]['name'], verbose = True)

    # -----------------------------
    # 2. Load datasets
    # -----------------------------
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=.5, hue=.3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ])

    train_dataset = SCUTDataset(
        root_dir = config["data"]["train_path"],    
        ann_path = config["data"]["train_ann"],
        processor = processor,
        train_transforms = train_transforms
    )

    val_dataset = SCUTDataset(
        root_dir = config["data"]["val_path"],     
        ann_path = config["data"]["val_ann"],
        processor = processor,
        train_transforms = None
    )
    # -----------------------------
    # 3. DataLoaders
    # -----------------------------
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size= config["dataloader"]["train_batch_size"], 
        shuffle=config["dataloader"]["shuffle"],
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size= config["dataloader"]["val_batch_size"], 
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=True,
        persistent_workers=True
    )

    

if __name__ == "__main__":
    config = ""
    main(config)
