import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# =========================================================
# 1. DATASET MULTIMODAL
# =========================================================
class MultimodalDataset(Dataset):
    def __init__(self, csv_file, img_dir, classes, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Image
        labels_str = row["Finding Labels"]
        diseases = labels_str.split('|') if pd.notna(labels_str) else []
        primary_class = diseases[0] if diseases else "Unknown"
        img_path = os.path.join(self.img_dir, primary_class, row["Image Index"])
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, row["Image Index"])
        
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Métadonnées (La 2ème modalité)
        gender = 1.0 if row["Patient Gender"] == "F" else 0.0
        view = 1.0 if row["View Position"] == "AP" else 0.0
        age = float(row["Patient Age"]) / 100.0
        meta_tensor = torch.tensor([gender, view, age], dtype=torch.float32)

        # Labels
        label = torch.zeros(len(self.classes), dtype=torch.float32)
        for d in diseases:
            if d in self.classes:
                label[self.classes.index(d)] = 1.0

        return image, meta_tensor, label

# =========================================================
# 2. MODÈLE DE FUSION
# =========================================================
class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalFusionModel, self).__init__()
        self.image_branch = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_ftrs = self.image_branch.classifier.in_features
        self.image_branch.classifier = nn.Identity() 
        
        self.meta_branch = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.classifier = nn.Linear(num_ftrs + 16, num_classes)

    def forward(self, img, meta):
        img_feats = self.image_branch(img)
        meta_feats = self.meta_branch(meta)
        combined = torch.cat((img_feats, meta_feats), dim=1)
        return self.classifier(combined)

# =========================================================
# 3. ENTRAÎNEMENT AVEC MLFLOW
# =========================================================
def train_multimodal():
    # Config MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Multimodal_Fusion_NIH")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "../output_split"
    batch_size = 32
    epochs = 10
    lr = 1e-4

    classes = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
               'Emphysema','Fibrosis','Infiltration','Mass','Nodule',
               'Pleural_Thickening','Pneumonia','Pneumothorax']

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = MultimodalDataset(os.path.join(data_dir, "train_metadata.csv"), os.path.join(data_dir, "train"), classes, transform)
    val_ds = MultimodalDataset(os.path.join(data_dir, "val_metadata.csv"), os.path.join(data_dir, "val"), classes, transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4)

    model = MultimodalFusionModel(len(classes)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    with mlflow.start_run(run_name="LateFusion_DenseNet_Meta"):
        # Log des hyperparamètres
        mlflow.log_param("model_architecture", "DenseNet121 + MLP")
        mlflow.log_param("fusion_type", "Late Fusion (Concatenation)")
        mlflow.log_param("lr", lr)
        mlflow.log_param("epochs", epochs)

        for epoch in range(epochs):
            # --- TRAIN ---
            model.train()
            train_loss = 0
            for imgs, metas, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs, metas)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            
            # --- VAL ---
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for imgs, metas, labels in val_loader:
                    imgs, metas, labels = imgs.to(device), metas.to(device), labels.to(device)
                    outputs = model(imgs, metas)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)

            # Log des métriques par epoch
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

            print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")

            # Sauvegarde du meilleur modèle
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "model_multimodal_best.pth")
                mlflow.pytorch.log_model(model, "best_multimodal_model")

        print("✅ Entraînement terminé et loggé dans MLflow.")

if __name__ == "__main__":
    train_multimodal()