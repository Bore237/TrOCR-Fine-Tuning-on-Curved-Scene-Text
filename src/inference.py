import os
import yaml
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from src import load_model 


def run_inference_on_image(model, processor, device, image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text


def run_inference_on_folder(model, processor, device, folder_path, save_csv=None):
    results = []

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for img_name in tqdm(image_files, desc="Inference"):
        img_path = os.path.join(folder_path, img_name)
        text = run_inference_on_image(model, processor, device, img_path)
        results.append({"image": img_name, "prediction": text})

    if save_csv:
        df = pd.DataFrame(results)
        df.to_csv(save_csv, index=False)

    return results


def main():
    # -----------------------------
    # Charger config.yaml
    # -----------------------------
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # -----------------------------
    # Charger modèle + processor
    # -----------------------------
    processor, model, device = load_model(config["model"]["name"])

    # -----------------------------
    # Mode : image ou dossier ?
    # -----------------------------
    mode = input("Mode (image/folder) : ").strip().lower()

    if mode == "image":
        img_path = input("Chemin de l'image : ").strip()
        text = run_inference_on_image(model, processor, device, img_path)
        print("\nPrediction :", text)

    elif mode == "folder":
        folder = input("Chemin du dossier : ").strip()
        save_csv = input("Sauvegarder CSV ? (chemin ou vide) : ").strip()
        save_csv = save_csv if save_csv else None

        results = run_inference_on_folder(
            model, processor, device, folder, save_csv
        )

        print("\nTerminé. Nombre d'images traitées :", len(results))

    else:
        print("Mode inconnu. Choisir 'image' ou 'folder'.")


if __name__ == "__main__":
    main()
