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
* Extraire les crops de texte
    * Lire les polygones annotés
    * decouper la zone correxpondante
    * renvoyer un crop + transcription
* Appliquer les transformations (Data augmentation)

## Desscription du dossier notebooks
Pour exploration et le test rapide

## Desscription du dossier configs





