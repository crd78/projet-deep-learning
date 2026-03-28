import torch
import numpy as np
from sklearn.metrics import classification_report
from utils import get_dataloaders, SimpleCNN
from train import get_model  # On importe get_model pour ne pas réécrire l'architecture
from tqdm import tqdm

def evaluate_model(model_path, model_type='scratch'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
               'Emphysema','Fibrosis','Infiltration','Mass','Nodule',
               'Pleural_Thickening','Pneumonia','Pneumothorax']
    
    # 1. Charger les données
    _, _, test_loader = get_dataloaders(batch_size=32)
    
    # 2. Configurer le modèle selon le type
    num_classes = len(classes)
    
    # On utilise la fonction de ton train.py pour être sûr d'avoir la même structure
    model = get_model(model_type, num_classes, device)
    
    # Chargement des poids
    print(f"Chargement des poids depuis {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"Évaluation du modèle : {model_type} sur {device}...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inférence"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # On garde le seuil à 0.2 comme tu as testé
            preds = torch.sigmoid(outputs) > 0.2
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    print(f"\n--- Rapport de Classification ({model_type.upper()}) ---")
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

if __name__ == "__main__":
   
    # evaluate_model("model_resnet_best.pth", model_type='resnet')
    evaluate_model("model_densenet_best.pth", model_type='densenet')