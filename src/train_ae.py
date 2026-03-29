import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import mlflow
from utils import MultiLabelDataset # On réutilise ton Dataset

# 1. Définition de l'Auto-encodeur Convolutionnel
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encodeur : Compresse l'image (224x224 -> 28x28)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        # Décodeur : Reconstruit l'image (28x28 -> 224x224)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Pour rester entre 0 et 1 (comme ToTensor)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_anomaly_detector():
    # --- Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "../output_split"
    batch_size = 16 # L'AE consomme pas mal de mémoire
    epochs = 10
    lr = 1e-3
    img_size = 224
    
    # --- Préparation des données (Uniquement "No Finding") ---
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # On charge le CSV et on filtre pour ne garder que les poumons SAINS
    full_df = pd.read_csv(os.path.join(data_dir, "train_metadata.csv"))
    normal_df = full_df[full_df["Finding Labels"] == "No Finding"]
    normal_df.to_csv(os.path.join(data_dir, "train_normal_only.csv"), index=False)

    train_dataset = MultiLabelDataset(
        os.path.join(data_dir, "train_normal_only.csv"),
        os.path.join(data_dir, "train"),
        classes=["No Finding"], # On s'en fiche un peu ici
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- Initialisation ---
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("Anomaly_Detection_AE")
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "Convolutional_AE")
        mlflow.log_param("epochs", epochs)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images, _ in loop:
                images = images.to(device)
                
                # Forward : on compare l'output à l'input original
                outputs = model(images)
                loss = criterion(outputs, images)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                loop.set_postfix(mse_loss=loss.item())

            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("mse_loss", avg_loss, step=epoch)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Sauvegarde du "détecteur d'anomalies"
        torch.save(model.state_dict(), "model_ae_anomaly.pth")
        print("Modèle Auto-encodeur sauvegardé sous 'model_ae_anomaly.pth'")

if __name__ == "__main__":
    train_anomaly_detector()