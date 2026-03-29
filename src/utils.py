import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# On sort SimpleCNN ici pour qu'il soit importable par evaluate.py et api.py
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
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
        diseases = row["Finding Labels"].split('|') if pd.notna(row["Finding Labels"]) else []
        
        primary_class = diseases[0] if diseases else "Unknown"
        img_path = os.path.join(self.img_dir, primary_class, row["Image Index"])
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, row["Image Index"])

        image = Image.open(img_path).convert("RGB")
        label = torch.zeros(len(self.classes), dtype=torch.float32)
        for d in diseases:
            if d in self.classes:
                label[self.classes.index(d)] = 1.0

        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(data_dir="../output_split", batch_size=32, img_size=224):
    classes = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
               'Emphysema','Fibrosis','Infiltration','Mass','Nodule',
               'Pleural_Thickening','Pneumonia','Pneumothorax']
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # On utilise le val_metadata pour l'évaluation
    test_dataset = MultiLabelDataset(os.path.join(data_dir, "val_metadata.csv"),
                                    os.path.join(data_dir, "val"), classes, transform)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return None, None, test_loader
