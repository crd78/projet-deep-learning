import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from utils import get_dataloaders
from train import get_model
import pandas as pd
from tqdm import tqdm  # Importation de tqdm

def generate_visuals(model_path, model_type='densenet'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
               'Emphysema','Fibrosis','Infiltration','Mass','Nodule',
               'Pleural_Thickening','Pneumonia','Pneumothorax']
    
    # 1. Charger le modèle et les données
    # Assure-tu que get_dataloaders pointe bien vers ton dossier "test"
    _, _, test_loader = get_dataloaders(batch_size=32)
    model = get_model(model_type, len(classes), device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_probs = []
    all_labels = []

    print(f"🚀 Extraction des prédictions sur {device}...")
    
    # Ajout de tqdm ici pour voir la progression
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inférence Test"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    # --- 2. MATRICE DE CONFUSION ---
    print("📊 Génération de la matrice de confusion...")
    y_true_idx = np.argmax(all_labels, axis=1)
    y_pred_idx = np.argmax(all_probs, axis=1)
    
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Matrice de Confusion - {model_type}')
    plt.ylabel('Vraie Pathologie')
    plt.xlabel('Pathologie Prédite')
    plt.tight_layout()
    plt.savefig('matrice_confusion.png')
    print("✅ Sauvegardée : matrice_confusion.png")

    # --- 3. COURBES ROC ---
    print("📈 Génération des courbes ROC...")
    plt.figure(figsize=(10, 8))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(all_labels[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title('Courbes ROC par Pathologie')
    plt.legend(loc="lower right", fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('courbes_roc.png')
    print("✅ Sauvegardée : courbes_roc.png")

if __name__ == "__main__":
    # Vérifie bien que ce fichier existe avant de lancer
    path_to_model = "model_densenet_best.pth"
    generate_visuals(path_to_model, model_type="densenet")