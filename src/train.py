import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def main():
    # 1. Configurations
    data_dir = "../data_train/kaggle/working/data"
    batch_size = 32
    num_epochs = 10
    img_size = 224
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 2. Transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # <-- AJOUTE CETTE LIGNE
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) 
    ])

    # 3. Datasets & Loaders
    # Assure-toi que les chemins existent pour éviter une autre erreur
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)

    # 4. Modèle (Correction de la dépréciation)
    # On utilise weights au lieu de pretrained
    weights = models.ResNet18_Weights.DEFAULT 
    model = models.resnet18(weights=weights)

    # On NE TOUCHE PAS à model.conv1 si on est en RGB
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Adaptation pour le niveaux de gris (1 canal au lieu de 3)
    if model.conv1.in_channels == 3:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 5. Optimiseur et perte
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. Boucle d'entraînement
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # 7. Sauvegarde du modèle
    torch.save(model.state_dict(), "model.pth")
    print("Modèle sauvegardé sous model.pth")

# C'EST CETTE LIGNE QUI MANQUAIT :
if __name__ == "__main__":
    main()