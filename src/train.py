import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import argparse
import mlflow
import mlflow.pytorch
from tqdm import tqdm  
import matplotlib.pyplot as plt

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
except ImportError:
    vit_b_16 = None
    ViT_B_16_Weights = None


# =========================================================
# DATASET MULTI-LABEL
# =========================================================
class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, img_dir, classes, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        labels_str = row["Finding Labels"]
        diseases = labels_str.split('|') if pd.notna(labels_str) else []

        # CORRECTION : Recherche dans le sous-dossier de la classe principale
        primary_class = diseases[0] if diseases else "Unknown"
        img_path = os.path.join(self.img_dir, primary_class, row["Image Index"])
        
        if not os.path.exists(img_path):
            # Fallback si l'image n'est pas dans un sous-dossier (dépend du split)
            img_path = os.path.join(self.img_dir, row["Image Index"])

        image = Image.open(img_path).convert("RGB")
        label = torch.zeros(len(self.classes), dtype=torch.float32)
        for d in diseases:
            if d in self.classes:
                label[self.classes.index(d)] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# MODELS
# =========================================================
def get_model(model_type, num_classes, device):
    if model_type == "scratch":
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 14 * 14, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
            def forward(self, x):
                return self.classifier(self.features(x))
        return SimpleCNN().to(device)

    elif model_type == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)

    elif model_type == "densenet":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model.to(device)

    elif model_type == "vit":
        if vit_b_16 is None:
            raise ImportError("ViT nécessite torchvision >= 0.13")
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model.to(device)
    else:
        raise ValueError("Modèle inconnu")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["scratch", "resnet", "densenet", "vit"])
    args = parser.parse_args()

   
    # 1. On définit où la base de données se trouve
    db_path = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(db_path)
    
    # 2. On définit l'expérience
    mlflow.set_experiment("Radiographie_Classification")
    
    # Configuration des dossiers
    data_dir = "../output_split"
    batch_size = 32
    num_epochs = 10
    lr = 1e-4
    img_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes = [
        'Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
        'Emphysema','Fibrosis','Infiltration','Mass','Nodule',
        'Pleural_Thickening','Pneumonia','Pneumothorax'
    ]

    # Utilisation d'un context manager pour le run
    with mlflow.start_run(run_name=f"Run_{args.model}"):
        # On log l'ID du run pour debug si besoin
        run_id = mlflow.active_run().info.run_id
        print(f"Lancement du Run ID: {run_id}")
        
        mlflow.log_param("model_type", args.model)
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        train_dataset = MultiLabelDataset(os.path.join(data_dir, "train_metadata.csv"),
                                         os.path.join(data_dir, "train"), classes, transform)
        val_dataset = MultiLabelDataset(os.path.join(data_dir, "val_metadata.csv"),
                                       os.path.join(data_dir, "val"), classes, transform)

        # AJOUT : num_workers=4 et pin_memory pour la vitesse
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=4, pin_memory=True)

        model = get_model(args.model, len(classes), device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        best_val_loss = float('inf')

        all_train_losses = []
        all_val_losses = []
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # --- PHASE TRAIN ---
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            
            for imgs, labels in train_pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / len(train_loader)
            all_train_losses.append(avg_train_loss) # AJOUT : on stocke la valeur

            # --- PHASE VAL ---
            model.eval()
            val_loss = 0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            
            with torch.no_grad():
                for imgs, labels in val_pbar:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            avg_val_loss = val_loss / len(val_loader)
            all_val_losses.append(avg_val_loss) # AJOUT : on stocke la valeur

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                mlflow.pytorch.log_model(model, "best_model")
                torch.save(model.state_dict(), f"model_{args.model}_best.pth")

        # =========================================================
        # ARTEFACTS (APRES LA BOUCLE EPOCH)
        # =========================================================
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), all_train_losses, label='Train Loss', marker='o')
        plt.plot(range(1, num_epochs + 1), all_val_losses, label='Val Loss', marker='x')
        plt.title(f"Courbe de Loss - {args.model}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # On sauvegarde l'image localement puis on l'envoie à MLflow
        plot_path = f"loss_curve_{args.model}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path) 
        
        print(f"Entraînement terminé. Graphique sauvegardé et loggé dans MLflow.")

if __name__ == "__main__":
    main()