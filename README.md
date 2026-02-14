# Fine-tuning TrOCR on SCUT-CTW1500

## Objectif
Fine-tuner TrOCR pour la reconnaissance de texte courbe sur le dataset SCUT-CTW1500.

## Workflow
- Développement local (Git)
- Entraînement sur Kaggle

## Desscription du dossier src
Ou sera stoker le code propre et versionné, reutilisable

### dataset.py
* Charger les images
* Charger les annotations depuis les fichiers text
* construire un dictionnaire :
* Appliquer les transformations (Data augmentation)
```
{ "image_path": "scut_train/006063.jpg", "text": "COLLEGE" }
```

## Desscription du dossier notebooks
Pour exploration et le test rapide

## Desscription du dossier configs





