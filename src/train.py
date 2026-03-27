import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import argparse

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

    # ---------------- CNN FROM SCRATCH ----------------
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

    # ---------------- RESNET (TRANSFER LEARNING) ----------------
    elif model_type == "resnet":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(device)

    # ---------------- DENSENET (TRANSFER LEARNING) ----------------
    elif model_type == "densenet":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model.to(device)

    # ---------------- VISION TRANSFORMER ----------------
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

    data_dir = "../output_split"
    batch_size = 32
    num_epochs = 10
    lr = 1e-4
    img_size = 224

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Model:", args.model)

    classes = [
        'Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
        'Emphysema','Fibrosis','Infiltration','Mass','Nodule',
        'Pleural_Thickening','Pneumonia','Pneumothorax'
    ]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_dataset = MultiLabelDataset(
        os.path.join(data_dir, "train_metadata.csv"),
        os.path.join(data_dir, "train"),
        classes,
        transform
    )

    val_dataset = MultiLabelDataset(
        os.path.join(data_dir, "val_metadata.csv"),
        os.path.join(data_dir, "val"),
        classes,
        transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(args.model, len(classes), device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # =========================================================
    # TRAIN
    # =========================================================
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), f"model_{args.model}.pth")
    print("Model saved")


if __name__ == "__main__":
    main()