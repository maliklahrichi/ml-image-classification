# 🧠 SM604 — Mathématiques pour le Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![NumPy](https://img.shields.io/badge/NumPy-used-lightgrey)
![EFREI](https://img.shields.io/badge/EFREI_Paris-2025--2026-purple)

> De la classification des chiffres manuscrits à la détection de cancers du sein.

Projet académique réalisé à **EFREI Paris** (SM604 — Semestre 6, Promotion 2028) implémentant des méthodes d'apprentissage supervisé **from scratch** (NumPy) et avec **PyTorch**, sur des problèmes de classification d'images de difficulté croissante.

## Parties du projet

### Partie 1 — MNIST (chiffres manuscrits)
- Modèle linéaire softmax avec descente de gradient mini-batch
- MLP à rétropropagation manuelle (1 et 2 couches cachées ReLU)
- Analyse des erreurs, matrice de confusion, visualisation PCA 2D

| Modèle | Paramètres | Erreur test |
|---|---|---|
| Linéaire softmax | 7 850 | ~7.7% |
| MLP 1 couche (128) | ~10 170 | ~3.5% |
| MLP 2 couches (128, 64) | ~10 938 | ~3.3% |

### Partie 2 — CIFAR-10 (images naturelles)
- Étude des filtres de convolution (Sobel, Prewitt, Moyenne, Emboss...)
- CNN PyTorch : Conv×4 → MaxPool×2 → FC256 → 10 classes
- Comparaison MLP vs CNN : ~820k params / 46% acc → ~189k params / 56% acc

### Partie 3 — BreastMNIST (diagnostic médical)
- Classification binaire malin/bénin sur échographies mammaires
- Gestion du déséquilibre de classes (pondération cross-entropy)
- Métriques médicales : AUC=0.856, Sensibilité=59.5%, Spécificité=92.1%
- Discussion éthique : calibration, explicabilité (Grad-CAM), réglementation

## Stack technique

- **NumPy** — implémentation from scratch (softmax, backprop, cross-entropy)
- **PyTorch + torchvision** — CNN, DataLoader, Autograd
- **scikit-learn** — métriques, PCA, t-SNE
- **matplotlib + seaborn** — visualisations

## Lancer le projet

```bash
pip install torch torchvision matplotlib seaborn scikit-learn pillow tqdm
# Notebook interactif
jupyter notebook Projet_SM604_Complet.ipynb
# Script Python standalone
python projet_sm604_complet.py
```

## Auteurs

Projet réalisé en équipe à **EFREI Paris** — SM604 Mathématiques pour le Machine Learning (2025-2026).
