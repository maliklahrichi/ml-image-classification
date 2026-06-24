#!/usr/bin/env python3
"""
=============================================================================
PROJET SM604 - Mathématiques pour le Machine Learning
De la classification des chiffres manuscrits à la détection de cancers du sein
EFREI Paris - Année 2025-2026 / Semestre 6
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os, time, warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 150, 'font.size': 11, 'axes.titlesize': 14, 'axes.titleweight': 'bold'})
PALETTE = ['#20808D', '#A84B2F', '#1B474D', '#BCE2E7', '#944454', '#FFC553', '#848456', '#6E522B']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
os.makedirs('figures', exist_ok=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================
def softmax(o):
    o_shifted = o - np.max(o, axis=1, keepdims=True)
    exp_o = np.exp(o_shifted)
    return exp_o / np.sum(exp_o, axis=1, keepdims=True)

def cross_entropy_loss(Y_true, P_pred, eps=1e-12):
    n = Y_true.shape[0]
    return -np.sum(Y_true * np.log(np.clip(P_pred, eps, 1-eps))) / n

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def one_hot_encode(y, num_classes=10):
    Y = np.zeros((len(y), num_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y

def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

# ============================================================================
# PARTIE 1 : MNIST
# ============================================================================
print("\n" + "="*80)
print("PARTIE 1 : CLASSIFICATION MNIST")
print("="*80)

# 1.1 Chargement
print("\n--- 1.1 Chargement MNIST ---")
transform_mnist = transforms.Compose([transforms.ToTensor()])
train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
test_ds = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)

X_train_mnist = train_ds.data.numpy().astype(np.float32)
y_train_mnist = train_ds.targets.numpy()
X_test_mnist = test_ds.data.numpy().astype(np.float32)
y_test_mnist = test_ds.targets.numpy()

X_train_flat = X_train_mnist.reshape(-1, 784) / 255.0
X_test_flat = X_test_mnist.reshape(-1, 784) / 255.0
Y_train_oh = one_hot_encode(y_train_mnist)
Y_test_oh = one_hot_encode(y_test_mnist)

print(f"Train: {X_train_flat.shape}, Test: {X_test_flat.shape}")

# Visualisation exemples
fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
fig.suptitle("Exemples de chiffres manuscrits MNIST", fontweight='bold')
for i in range(10):
    for j in range(2):
        idx = np.where(y_train_mnist == i)[0][j]
        axes[j, i].imshow(X_train_mnist[idx], cmap='gray'); axes[j, i].axis('off')
        if j == 0: axes[j, i].set_title(str(i))
plt.tight_layout(); plt.savefig('figures/mnist_exemples.png', bbox_inches='tight'); plt.close()

# ============================================================================
# 1.2.1 Modèle linéaire
# ============================================================================
print("\n--- 1.2.1 Modèle linéaire ---")
print("o_k = Σ a_{k,j} x_j + a_{k,0} → P_k = softmax(o)_k")
print("L = -(1/n) Σ_i Σ_k y_i^(k) ln(P_k(x_i))")
print("∂L/∂A = (1/n)(P-Y)^T X,  ∂L/∂b = (1/n) Σ(P-Y)")

class LinearClassifier:
    def __init__(self, d=784, K=10):
        self.A = np.random.randn(K, d) * np.sqrt(2.0/d)
        self.b = np.zeros(K)
        self.history = {'loss_tr':[], 'loss_te':[], 'acc_tr':[], 'acc_te':[]}
    
    def forward(self, X): return softmax(X @ self.A.T + self.b)
    def predict(self, X): return np.argmax(self.forward(X), axis=1)
    
    def train(self, Xtr, Ytr, ytr, Xte, Yte, yte, lr=0.5, epochs=30, bs=512):
        n = Xtr.shape[0]
        for ep in range(epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, bs):
                e = min(s+bs, n)
                Xb, Yb = Xtr[idx[s:e]], Ytr[idx[s:e]]
                Pb = self.forward(Xb)
                delta = Pb - Yb
                self.A -= lr * (delta.T @ Xb) / (e-s)
                self.b -= lr * np.mean(delta, axis=0)
            Ptr = self.forward(Xtr); Pte = self.forward(Xte)
            self.history['loss_tr'].append(cross_entropy_loss(Ytr, Ptr))
            self.history['loss_te'].append(cross_entropy_loss(Yte, Pte))
            self.history['acc_tr'].append(accuracy(ytr, np.argmax(Ptr,1)))
            self.history['acc_te'].append(accuracy(yte, np.argmax(Pte,1)))
            if (ep+1) % 10 == 0:
                print(f"  Ep {ep+1:3d} | Loss: {self.history['loss_tr'][-1]:.4f}/{self.history['loss_te'][-1]:.4f} | Acc: {self.history['acc_tr'][-1]*100:.2f}%/{self.history['acc_te'][-1]*100:.2f}%")

model_lin = LinearClassifier(784, 10)
model_lin.train(X_train_flat, Y_train_oh, y_train_mnist, X_test_flat, Y_test_oh, y_test_mnist, lr=0.5, epochs=30, bs=512)
err_lin = 1 - model_lin.history['acc_te'][-1]
print(f"\nTaux d'erreur test linéaire : {err_lin*100:.2f}%  (params: 7850)")

# ============================================================================
# 1.2.2 MLP avec rétropropagation
# ============================================================================
print("\n--- 1.2.2 MLP avec couches cachées ---")

class MLPClassifier:
    """
    Rétropropagation :
    δ^(last) = P - Y
    δ^h = (δ^(h+1) W^(h+1)) ⊙ φ'(o^h)
    W^h -= lr/n * δ^h^T z^(h-1)
    b^h -= lr/n * Σ δ^h
    """
    def __init__(self, dims):
        self.L = len(dims)-1
        self.W = [np.random.randn(dims[i+1], dims[i]) * np.sqrt(2.0/dims[i]) for i in range(self.L)]
        self.b = [np.zeros(dims[i+1]) for i in range(self.L)]
        self.history = {'loss_tr':[], 'loss_te':[], 'acc_tr':[], 'acc_te':[]}
    
    def forward(self, X):
        self.z = [X]; self.o = []
        cur = X
        for i in range(self.L):
            oi = cur @ self.W[i].T + self.b[i]
            self.o.append(oi)
            cur = relu(oi) if i < self.L-1 else softmax(oi)
            self.z.append(cur)
        return cur
    
    def backward(self, Y, lr):
        n = Y.shape[0]; delta = self.z[-1] - Y
        for i in range(self.L-1, -1, -1):
            dW = (delta.T @ self.z[i]) / n
            db = np.mean(delta, axis=0)
            if i > 0: delta = (delta @ self.W[i]) * relu_derivative(self.o[i-1])
            self.W[i] -= lr * dW; self.b[i] -= lr * db
    
    def predict(self, X): return np.argmax(self.forward(X), axis=1)
    def count_params(self): return sum(w.size + b.size for w, b in zip(self.W, self.b))
    
    def train(self, Xtr, Ytr, ytr, Xte, Yte, yte, lr=0.1, epochs=30, bs=512):
        n = Xtr.shape[0]
        for ep in range(epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, bs):
                e = min(s+bs, n)
                self.forward(Xtr[idx[s:e]]); self.backward(Ytr[idx[s:e]], lr)
            Ptr = self.forward(Xtr); Pte = self.forward(Xte)
            self.history['loss_tr'].append(cross_entropy_loss(Ytr, Ptr))
            self.history['loss_te'].append(cross_entropy_loss(Yte, Pte))
            self.history['acc_tr'].append(accuracy(ytr, np.argmax(Ptr,1)))
            self.history['acc_te'].append(accuracy(yte, np.argmax(Pte,1)))
            if (ep+1) % 10 == 0:
                print(f"  Ep {ep+1:3d} | Acc: {self.history['acc_tr'][-1]*100:.2f}%/{self.history['acc_te'][-1]*100:.2f}%")

# 1 couche cachée
print("\n>> MLP 1 couche : 784 → 128 (ReLU) → 10 (softmax)")
m1 = MLPClassifier([784, 128, 10])
m1.train(X_train_flat, Y_train_oh, y_train_mnist, X_test_flat, Y_test_oh, y_test_mnist, lr=0.1, epochs=30, bs=512)
err_1l = 1 - m1.history['acc_te'][-1]
print(f"  Err test: {err_1l*100:.2f}% | Params: {m1.count_params()}")

# 2 couches cachées
print("\n>> MLP 2 couches : 784 → 128 (ReLU) → 64 (ReLU) → 10 (softmax)")
m2 = MLPClassifier([784, 128, 64, 10])
m2.train(X_train_flat, Y_train_oh, y_train_mnist, X_test_flat, Y_test_oh, y_test_mnist, lr=0.1, epochs=30, bs=512)
err_2l = 1 - m2.history['acc_te'][-1]
print(f"  Err test: {err_2l*100:.2f}% | Params: {m2.count_params()}")

# Courbes comparatives
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Comparaison modèles MNIST", fontweight='bold')
for idx, (name, mdl) in enumerate([("Linéaire", model_lin), ("1 couche (128)", m1), ("2 couches (128,64)", m2)]):
    axes[idx].plot([a*100 for a in mdl.history['acc_tr']], label='Train', color=PALETTE[0], lw=2)
    axes[idx].plot([a*100 for a in mdl.history['acc_te']], label='Test', color=PALETTE[1], lw=2)
    axes[idx].set_xlabel('Époque'); axes[idx].set_ylabel('Précision (%)'); axes[idx].set_title(name)
    axes[idx].legend(); axes[idx].set_ylim([85, 100]); axes[idx].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('figures/mnist_comparaison.png', bbox_inches='tight'); plt.close()

# ============================================================================
# 1.2.3 Analyse des erreurs
# ============================================================================
print("\n--- 1.2.3 Analyse des erreurs ---")
best_pred = m2.predict(X_test_flat)
cm = confusion_matrix(y_test_mnist, best_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=range(10), yticklabels=range(10))
ax.set_xlabel('Prédiction'); ax.set_ylabel('Vérité')
ax.set_title('Matrice de confusion - MLP 2 couches (MNIST)', fontweight='bold')
plt.tight_layout(); plt.savefig('figures/mnist_confusion.png', bbox_inches='tight'); plt.close()

# Chiffres mal classés
mis = np.where(best_pred != y_test_mnist)[0]
print(f"Mal classés : {len(mis)} / {len(y_test_mnist)}")
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
fig.suptitle("Exemples de chiffres mal classés", fontweight='bold')
for i in range(min(20, len(mis))):
    ax = axes[i//10, i%10]; ax.imshow(X_test_mnist[mis[i]], cmap='gray')
    ax.set_title(f"V:{y_test_mnist[mis[i]]} P:{best_pred[mis[i]]}", fontsize=8, color='red'); ax.axis('off')
plt.tight_layout(); plt.savefig('figures/mnist_mal_classes.png', bbox_inches='tight'); plt.close()

# PCA 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_flat)
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test_mnist, cmap='tab10', s=3, alpha=0.4)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('Projection PCA 2D - MNIST', fontweight='bold')
plt.colorbar(sc, ax=ax, label='Classe')
plt.tight_layout(); plt.savefig('figures/mnist_pca.png', bbox_inches='tight'); plt.close()

print("\n" + "="*70)
print("RÉCAPITULATIF MNIST")
print("="*70)
print(f"{'Modèle':<25} {'Params':>8} {'Err. Train':>12} {'Err. Test':>12}")
print("-"*60)
print(f"{'Linéaire':<25} {7850:>8} {(1-model_lin.history['acc_tr'][-1])*100:>11.2f}% {err_lin*100:>11.2f}%")
print(f"{'1 couche (128)':<25} {m1.count_params():>8} {(1-m1.history['acc_tr'][-1])*100:>11.2f}% {err_1l*100:>11.2f}%")
print(f"{'2 couches (128,64)':<25} {m2.count_params():>8} {(1-m2.history['acc_tr'][-1])*100:>11.2f}% {err_2l*100:>11.2f}%")
print("="*70)

# ============================================================================
# PARTIE 2 : CIFAR-10
# ============================================================================
print("\n" + "="*80)
print("PARTIE 2 : CIFAR-10")
print("="*80)

train_ds_c = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_ds_c = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
X_tr_c = np.array(train_ds_c.data, dtype=np.float32)
y_tr_c = np.array(train_ds_c.targets)
X_te_c = np.array(test_ds_c.data, dtype=np.float32)
y_te_c = np.array(test_ds_c.targets)
CLASSES = ['avion','auto','oiseau','chat','cerf','chien','grenouille','cheval','bateau','camion']

# Exemples
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
fig.suptitle("Exemples CIFAR-10", fontweight='bold')
for i in range(10):
    for j in range(2):
        idx = np.where(y_tr_c == i)[0][j]
        axes[j, i].imshow(X_tr_c[idx].astype(np.uint8)); axes[j, i].axis('off')
        if j == 0: axes[j, i].set_title(CLASSES[i], fontsize=8)
plt.tight_layout(); plt.savefig('figures/cifar_exemples.png', bbox_inches='tight'); plt.close()

# 2.2 Préliminaire
print("\n--- 2.2 Travail préliminaire ---")
Ytr_c = one_hot_encode(y_tr_c); Yte_c = one_hot_encode(y_te_c)

# Niveaux de gris
Xtr_g = (0.299*X_tr_c[:,:,:,0] + 0.587*X_tr_c[:,:,:,1] + 0.114*X_tr_c[:,:,:,2]).reshape(-1,1024)/255.0
Xte_g = (0.299*X_te_c[:,:,:,0] + 0.587*X_te_c[:,:,:,1] + 0.114*X_te_c[:,:,:,2]).reshape(-1,1024)/255.0

# Sous-ensemble pour accélérer (10k train, 5k test)
sub_tr, sub_te = 10000, 5000
idx_tr = np.random.choice(len(Xtr_g), sub_tr, replace=False)
idx_te = np.random.choice(len(Xte_g), sub_te, replace=False)

print("\n>> Linéaire (gris) sur sous-ensemble...")
m_cg = LinearClassifier(1024, 10)
m_cg.train(Xtr_g[idx_tr], Ytr_c[idx_tr], y_tr_c[idx_tr], Xte_g[idx_te], Yte_c[idx_te], y_te_c[idx_te], lr=0.1, epochs=20, bs=512)
err_cg = 1 - m_cg.history['acc_te'][-1]
print(f"  Err test: {err_cg*100:.2f}%")

print("\n>> MLP (gris)...")
m_mg = MLPClassifier([1024, 256, 128, 10])
m_mg.train(Xtr_g[idx_tr], Ytr_c[idx_tr], y_tr_c[idx_tr], Xte_g[idx_te], Yte_c[idx_te], y_te_c[idx_te], lr=0.05, epochs=20, bs=512)
err_mg = 1 - m_mg.history['acc_te'][-1]
print(f"  Err test: {err_mg*100:.2f}%")

# Couleur
Xtr_col = X_tr_c.reshape(-1,3072)/255.0; Xte_col = X_te_c.reshape(-1,3072)/255.0

print("\n>> Linéaire (couleur)...")
m_cc = LinearClassifier(3072, 10)
m_cc.train(Xtr_col[idx_tr], Ytr_c[idx_tr], y_tr_c[idx_tr], Xte_col[idx_te], Yte_c[idx_te], y_te_c[idx_te], lr=0.05, epochs=20, bs=512)
err_cc = 1 - m_cc.history['acc_te'][-1]
print(f"  Err test: {err_cc*100:.2f}%")

print("\n>> MLP (couleur)...")
m_mc = MLPClassifier([3072, 256, 128, 10])
m_mc.train(Xtr_col[idx_tr], Ytr_c[idx_tr], y_tr_c[idx_tr], Xte_col[idx_te], Yte_c[idx_te], y_te_c[idx_te], lr=0.05, epochs=20, bs=512)
err_mc = 1 - m_mc.history['acc_te'][-1]
print(f"  Err test: {err_mc*100:.2f}%")

# 2.3 Filtres de convolution
print("\n--- 2.3 Filtres de convolution ---")
sample = (0.299*X_tr_c[np.where(y_tr_c==3)[0][0]][:,:,0] + 0.587*X_tr_c[np.where(y_tr_c==3)[0][0]][:,:,1] + 0.114*X_tr_c[np.where(y_tr_c==3)[0][0]][:,:,2])

K1 = np.ones((3,3))/9.0
K2 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
K3 = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])
K4 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
K5 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
K6 = np.array([[-2,-1,0],[-1,1,1],[0,1,2]])
filters = [K1,K2,K3,K4,K5,K6]
fnames = ['K1: Moyenne','K2: Netteté','K3: Bords vert.','K4: Prewitt','K5: Sobel','K6: Emboss']

def conv2d(img, K, l=0):
    h,w = img.shape; kh,kw = K.shape; ph,pw = kh//2, kw//2
    padded = np.pad(img, ((ph,ph),(pw,pw)), 'constant')
    out = np.zeros_like(img)
    for u in range(h):
        for v in range(w):
            out[u,v] = np.sum(padded[u:u+kh, v:v+kw] * K) + l
    return out

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Effet des filtres de convolution (l=0)", fontweight='bold')
axes[0,0].imshow(sample, cmap='gray'); axes[0,0].set_title('Originale'); axes[0,0].axis('off')
for i,(K,nm) in enumerate(zip(filters, fnames)):
    r,c = (i+1)//4, (i+1)%4
    axes[r,c].imshow(conv2d(sample, K), cmap='gray'); axes[r,c].set_title(nm, fontsize=9); axes[r,c].axis('off')
axes[1,3].axis('off')
plt.tight_layout(); plt.savefig('figures/cifar_filtres.png', bbox_inches='tight'); plt.close()

# 2.5-2.6 CNN PyTorch
print("\n--- 2.5/2.6 CNN CIFAR-10 (PyTorch) ---")

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1); self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1); self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1); self.bn4 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(8*8*64, 256); self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(self.pool1(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop(self.pool2(x))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(self.drop(x))

tf_tr = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
tf_te = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])

tr_cifar = torchvision.datasets.CIFAR10('./data', True, transform=tf_tr)
te_cifar = torchvision.datasets.CIFAR10('./data', False, transform=tf_te)
trl = DataLoader(tr_cifar, 128, True, num_workers=2)
tel = DataLoader(te_cifar, 128, False, num_workers=2)

cnn = CIFAR10_CNN().to(DEVICE)
nparams = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
print(f"Params CNN: {nparams:,}")
crit = nn.CrossEntropyLoss()
opt = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)
sched = optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)

cnn_h = {'loss_tr':[], 'loss_te':[], 'acc_tr':[], 'acc_te':[]}
EPOCHS = 15

for ep in range(EPOCHS):
    cnn.train(); rl, cor, tot = 0., 0, 0
    for inp, lab in trl:
        inp, lab = inp.to(DEVICE), lab.to(DEVICE)
        opt.zero_grad(); out = cnn(inp); loss = crit(out, lab); loss.backward(); opt.step()
        rl += loss.item()*inp.size(0); _, pred = out.max(1); tot += lab.size(0); cor += pred.eq(lab).sum().item()
    cnn_h['loss_tr'].append(rl/tot); cnn_h['acc_tr'].append(cor/tot)
    
    cnn.eval(); rl, cor, tot = 0., 0, 0
    with torch.no_grad():
        for inp, lab in tel:
            inp, lab = inp.to(DEVICE), lab.to(DEVICE)
            out = cnn(inp); loss = crit(out, lab)
            rl += loss.item()*inp.size(0); _, pred = out.max(1); tot += lab.size(0); cor += pred.eq(lab).sum().item()
    cnn_h['loss_te'].append(rl/tot); cnn_h['acc_te'].append(cor/tot)
    sched.step()
    if (ep+1) % 5 == 0:
        print(f"  Ep {ep+1}/{EPOCHS} | Acc: {cnn_h['acc_tr'][-1]*100:.2f}%/{cnn_h['acc_te'][-1]*100:.2f}%")

err_cnn = 1 - cnn_h['acc_te'][-1]

# Courbes CNN
fig, (a1,a2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CNN CIFAR-10", fontweight='bold')
a1.plot(cnn_h['loss_tr'], label='Train', color=PALETTE[0], lw=2)
a1.plot(cnn_h['loss_te'], label='Test', color=PALETTE[1], lw=2)
a1.set_xlabel('Époque'); a1.set_ylabel('Loss'); a1.set_title('Coût'); a1.legend()
a2.plot([a*100 for a in cnn_h['acc_tr']], label='Train', color=PALETTE[0], lw=2)
a2.plot([a*100 for a in cnn_h['acc_te']], label='Test', color=PALETTE[1], lw=2)
a2.set_xlabel('Époque'); a2.set_ylabel('Précision (%)'); a2.set_title('Précision'); a2.legend()
plt.tight_layout(); plt.savefig('figures/cifar_cnn_courbes.png', bbox_inches='tight'); plt.close()

# Confusion CIFAR
cnn.eval(); ap, al = [], []
with torch.no_grad():
    for inp, lab in tel:
        inp = inp.to(DEVICE); _, pred = cnn(inp).max(1)
        ap.extend(pred.cpu().numpy()); al.extend(lab.numpy())
cm_c = confusion_matrix(al, ap)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_c, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=CLASSES, yticklabels=CLASSES)
ax.set_xlabel('Prédiction'); ax.set_ylabel('Vérité')
ax.set_title('Matrice de confusion CNN - CIFAR-10', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.savefig('figures/cifar_confusion.png', bbox_inches='tight'); plt.close()

print(f"\n{'='*70}")
print("RÉCAPITULATIF CIFAR-10")
print(f"{'='*70}")
print(f"{'Modèle':<40} {'Err. Test':>12}")
print(f"{'-'*55}")
print(f"{'Linéaire (gris)':<40} {err_cg*100:>11.2f}%")
print(f"{'MLP 2 couches (gris)':<40} {err_mg*100:>11.2f}%")
print(f"{'Linéaire (couleur)':<40} {err_cc*100:>11.2f}%")
print(f"{'MLP 2 couches (couleur)':<40} {err_mc*100:>11.2f}%")
print(f"{'CNN (Conv64×4 + FC)':<40} {err_cnn*100:>11.2f}%")
print(f"{'─'*55}")
print(f"Réf: ConvDBN(2010)=21.1% | Maxout(2013)=9.4% | ViT(2021)=0.5%")
print(f"{'='*70}")

# ============================================================================
# PARTIE 3 : MAMMOGRAPHIES CBIS-DDSM
# ============================================================================
print("\n" + "="*80)
print("PARTIE 3 : DIAGNOSTIC MÉDICAL (CBIS-DDSM)")
print("="*80)

class MammoCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout(0.25),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout(0.25),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2,2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)), nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    def forward(self, x): return self.classifier(self.features(x))

# Demo avec données simulées
print("\nDémonstration avec données simulées (remplacer par CBIS-DDSM)")
n_b, n_m = 400, 160
X_sim = torch.randn(n_b+n_m, 1, 128, 128)
y_sim = torch.cat([torch.zeros(n_b, dtype=torch.long), torch.ones(n_m, dtype=torch.long)])
perm = torch.randperm(n_b+n_m); X_sim, y_sim = X_sim[perm], y_sim[perm]
sp = int(0.8*(n_b+n_m))
trl_m = DataLoader(TensorDataset(X_sim[:sp], y_sim[:sp]), 32, True)
tel_m = DataLoader(TensorDataset(X_sim[sp:], y_sim[sp:]), 32, False)

w0 = sp / (2*(y_sim[:sp]==0).sum().item()); w1 = sp / (2*(y_sim[:sp]==1).sum().item())
print(f"Poids classes: bénin={w0:.2f}, malin={w1:.2f}")

mam = MammoCNN().to(DEVICE)
mp = sum(p.numel() for p in mam.parameters() if p.requires_grad)
print(f"Params MammoCNN: {mp:,}")

crit_m = nn.CrossEntropyLoss(weight=torch.FloatTensor([w0, w1]).to(DEVICE))
opt_m = optim.Adam(mam.parameters(), lr=0.0001)

for ep in range(10):
    mam.train()
    for inp, lab in trl_m:
        inp, lab = inp.to(DEVICE), lab.to(DEVICE)
        opt_m.zero_grad(); loss = crit_m(mam(inp), lab); loss.backward(); opt_m.step()

mam.eval(); ap_m, al_m = [], []
with torch.no_grad():
    for inp, lab in tel_m:
        _, pred = mam(inp.to(DEVICE)).max(1)
        ap_m.extend(pred.cpu().numpy()); al_m.extend(lab.numpy())

cm_m = confusion_matrix(al_m, ap_m)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm_m, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Bénin','Malin'], yticklabels=['Bénin','Malin'])
ax.set_xlabel('Prédiction'); ax.set_ylabel('Vérité')
ax.set_title('Matrice de confusion - Mammographies', fontweight='bold')
plt.tight_layout(); plt.savefig('figures/mammo_confusion.png', bbox_inches='tight'); plt.close()

from sklearn.metrics import precision_score, recall_score, f1_score
prec = precision_score(al_m, ap_m, zero_division=0)
rec = recall_score(al_m, ap_m, zero_division=0)
f1 = f1_score(al_m, ap_m, zero_division=0)
print(f"Precision: {prec*100:.1f}% | Recall: {rec*100:.1f}% | F1: {f1*100:.1f}%")

# ============================================================================
# SYNTHÈSE
# ============================================================================
print("\n" + "="*80)
print("SYNTHÈSE FINALE")
print("="*80)
print(f"""
MNIST:
  Linéaire        → {err_lin*100:.2f}% erreur
  MLP 1 couche    → {err_1l*100:.2f}% erreur
  MLP 2 couches   → {err_2l*100:.2f}% erreur

CIFAR-10:
  Linéaire (gris)  → {err_cg*100:.2f}% erreur
  MLP (gris)       → {err_mg*100:.2f}% erreur
  Linéaire (color) → {err_cc*100:.2f}% erreur
  MLP (color)      → {err_mc*100:.2f}% erreur
  CNN              → {err_cnn*100:.2f}% erreur

Mammographies: Architecture prête (MammoCNN, {mp:,} params)
  → Remplacer données simulées par CBIS-DDSM

Figures sauvegardées dans ./figures/
""")
print("PROJET TERMINÉ")
