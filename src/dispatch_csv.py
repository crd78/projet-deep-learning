"""
Script de splitting du dataset avec stratification des classes
Conserve la proportion de chaque classe dans train/val/test
Compatible avec multi-label (plusieurs maladies par image)
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import json
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chemins
CSV_PATH = "../data_train/kaggle/working/data/train_meta_data.csv"
TRAIN_DATA_DIR = "../data_train/kaggle/working/data/train_data"
OUTPUT_BASE_DIR = "../data_train/kaggle/working/data"

# Proportions (doit sommer à 1.0)
TRAIN_RATIO = 0.70                            # 70% pour entraînement
VAL_RATIO = 0.15                              # 15% pour validation
TEST_RATIO = 0.15                             # 15% pour test

# Stratégie de stratification
STRATIFY_BY = "primary_disease"  # Options : 'primary_disease' ou 'all_diseases'
RANDOM_STATE = 42                  # Pour reproductibilité

# Créer les dossiers de sortie
CREATE_SUBDIRS = True             # Créer des sous-dossiers par classe
COPY_IMAGES = True                # Copier les images (False = créer des symlinks)

# ============================================================================
# FONCTIONS
# ============================================================================

def load_metadata(csv_path):
    """Charger et parser le CSV"""
    print(f"📖 Chargement du CSV : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ {len(df)} images chargées")
    print(f"Colonnes : {list(df.columns)}")
    return df

def parse_diseases(disease_str):
    """Parser la colonne Diseases (format string de liste)"""
    if pd.isna(disease_str):
        return []
    # Convertir le string en liste Python
    try:
        import ast
        return ast.literal_eval(disease_str)
    except:
        return []

def create_stratify_column(df, strategy="primary_disease"):
    """Créer une colonne pour la stratification"""
    if strategy == "primary_disease":
        # Utiliser la première maladie seulement
        df['stratify_col'] = df['Diseases'].apply(
            lambda x: parse_diseases(x)[0] if parse_diseases(x) else 'Unknown'
        )
    elif strategy == "all_diseases":
        # Utiliser toutes les maladies combinées
        df['stratify_col'] = df['Diseases'].apply(
            lambda x: '_'.join(sorted(parse_diseases(x))) if parse_diseases(x) else 'Unknown'
        )
    else:
        raise ValueError(f"Stratégie inconnue : {strategy}")
    
    return df

def check_distribution(df, label_column, title):
    """Afficher la distribution des classes"""
    print(f"\n📊 Distribution {title}:")
    counts = df[label_column].value_counts()
    percentages = (counts / len(df) * 100).round(2)
    
    for disease, count in counts.items():
        pct = percentages[disease]
        print(f"  {disease}: {count:5d} images ({pct:6.2f}%)")
    
    return counts, percentages

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Splitter les données avec stratification"""
    
    print("\n🔄 Stratification des données...")
    
    # Vérification des ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Les ratios doivent sommer à 1.0"
    
    # Split 1 : Séparer train du reste (val + test)
    rest_ratio = val_ratio + test_ratio
    train_df, rest_df = train_test_split(
        df,
        test_size=rest_ratio,
        train_size=train_ratio,
        stratify=df['stratify_col'],
        random_state=random_state
    )
    
    # Split 2 : Séparer val et test (avec même stratification)
    val_ratio_adjusted = val_ratio / rest_ratio
    val_df, test_df = train_test_split(
        rest_df,
        test_size=1 - val_ratio_adjusted,
        train_size=val_ratio_adjusted,
        stratify=rest_df['stratify_col'],
        random_state=random_state
    )
    
    print(f"\n✓ Split effectué:")
    print(f"  Train : {len(train_df)} images")
    print(f"  Val   : {len(val_df)} images")
    print(f"  Test  : {len(test_df)} images")
    
    return train_df, val_df, test_df

def create_directory_structure(base_dir, classes, subdirs=['train', 'val', 'test']):
    """Créer la structure des dossiers"""
    print(f"\n📁 Création des dossiers dans {base_dir}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        
        # Créer des sous-dossiers par classe
        for cls in classes:
            cls_dir = os.path.join(subdir_path, cls)
            os.makedirs(cls_dir, exist_ok=True)
    
    print("✓ Dossiers créés")

def copy_files(df, source_dir, dest_dir, subset_name, valid_classes, copy=True):
    """Copier ou créer des symlinks pour les images"""
    print(f"\n📸 Traitement du subset {subset_name.upper()} ({len(df)} images)...")
    
    success = 0
    errors = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_name = row['Image Index']
        source_path = os.path.join(source_dir, image_name)
        
        # Déterminer la classe (première maladie)
        diseases = parse_diseases(row['Diseases'])
        class_name = diseases[0] if diseases else 'Unknown'

        # skip si classe supprimée
        if class_name not in valid_classes:
            continue
        
        # Chemin de destination
        dest_path = os.path.join(dest_dir, class_name, image_name)
        
        try:
            if not os.path.exists(source_path):
                errors.append(f"Source not found: {source_path}")
                continue
            
            if copy:
                # Copier le fichier
                shutil.copy2(source_path, dest_path)
            else:
                # Créer un symlink (économe en espace)
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                os.symlink(os.path.abspath(source_path), dest_path)
            
            success += 1
        except Exception as e:
            errors.append(f"Error with {image_name}: {str(e)}")
    
    print(f"✓ {success}/{len(df)} images traitées")
    
    if errors:
        print(f"⚠️  {len(errors)} erreurs:")
        for err in errors[:5]:  # Afficher les 5 premières
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... et {len(errors) - 5} autres")
    
    return success, errors

def save_metadata(train_df, val_df, test_df, output_dir):
    """Sauvegarder les métadonnées de chaque split"""
    print("\n💾 Sauvegarde des métadonnées...")
    
    # Sauvegarder les CSVs
    train_df.to_csv(os.path.join(output_dir, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_metadata.csv'), index=False)
    
    print("✓ Métadonnées sauvegardées")

def generate_report(train_df, val_df, test_df, output_dir):
    """Générer un rapport de distribution"""
    print("\n📋 Génération du rapport...")
    
    report = {
        "total_images": len(train_df) + len(val_df) + len(test_df),
        "splits": {
            "train": {
                "count": len(train_df),
                "percentage": round(len(train_df) / (len(train_df) + len(val_df) + len(test_df)) * 100, 2)
            },
            "val": {
                "count": len(val_df),
                "percentage": round(len(val_df) / (len(train_df) + len(val_df) + len(test_df)) * 100, 2)
            },
            "test": {
                "count": len(test_df),
                "percentage": round(len(test_df) / (len(train_df) + len(val_df) + len(test_df)) * 100, 2)
            }
        },
        "class_distribution": {}
    }
    
    # Analyser la distribution par classe pour chaque split
    for subset_name, subset_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        report["class_distribution"][subset_name] = {}
        
        for idx, row in subset_df.iterrows():
            diseases = parse_diseases(row['Diseases'])
            class_name = diseases[0] if diseases else 'Unknown'
            
            if class_name not in report["class_distribution"][subset_name]:
                report["class_distribution"][subset_name][class_name] = 0
            report["class_distribution"][subset_name][class_name] += 1
    
    # Afficher le rapport
    print("\n" + "="*70)
    print("RAPPORT DE DISTRIBUTION DES CLASSES")
    print("="*70)
    
    print(f"\nTotal: {report['total_images']} images\n")
    
    for subset_name in ["train", "val", "test"]:
        count = report["splits"][subset_name]["count"]
        pct = report["splits"][subset_name]["percentage"]
        print(f"{subset_name.upper()}: {count} images ({pct}%)")
        
        classes = report["class_distribution"][subset_name]
        for cls, count in sorted(classes.items(), key=lambda x: x[1], reverse=True):
            pct_class = round(count / len(subset_df) * 100, 2) if len(subset_df) > 0 else 0
            print(f"  ├─ {cls}: {count:4d} ({pct_class:6.2f}%)")
        print()
    
    # Sauvegarder le rapport
    with open(os.path.join(output_dir, 'split_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✓ Rapport sauvegardé")
    print("="*70)

def main():
    """Fonction principale"""
    
    print("="*70)
    print("🔄 SPLITTING DU DATASET AVEC STRATIFICATION")
    print("="*70)
    
    # 1. Charger les données
    df = load_metadata(CSV_PATH)
    
    # 2. Créer la colonne de stratification
    df = create_stratify_column(df, strategy=STRATIFY_BY)
    
    # 3. Supprimer les images dont la catégorie principale a moins de 50 images
    min_images = 50
    class_counts = df['stratify_col'].value_counts()
    valid_classes = class_counts[class_counts >= min_images].index
    df = df[df['stratify_col'].isin(valid_classes)].copy()
    print(f"Catégories conservées (>=50 images): {list(valid_classes)}")
    print(f"Nombre d'images après filtrage: {len(df)}")

    # 4. Afficher la distribution originale
    print("\n" + "="*70)
    check_distribution(df, 'stratify_col', "ORIGINALE")
    print("="*70)
    
    # 4. Splitter les données
    train_df, val_df, test_df = split_data(
        df,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_state=RANDOM_STATE
    )
    
    # 5. Vérifier que la distribution est conservée
    print("\n" + "="*70)
    print("VÉRIFICATION DE LA STRATIFICATION")
    print("="*70)
    check_distribution(train_df, 'stratify_col', "TRAIN")
    check_distribution(val_df, 'stratify_col', "VALIDATION")
    check_distribution(test_df, 'stratify_col', "TEST")
    print("="*70)

    # Afficher la distribution exacte par split pour debug
    for name, subset_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\nDistribution brute pour {name}:")
        print(subset_df['stratify_col'].value_counts())
    
    # 6. Créer la structure des dossiers
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    if CREATE_SUBDIRS:
        unique_classes = sorted(valid_classes)
        create_directory_structure(OUTPUT_BASE_DIR, unique_classes)
    
    # 7. Copier les fichiers
    if COPY_IMAGES:
        copy_files(train_df, TRAIN_DATA_DIR, os.path.join(OUTPUT_BASE_DIR, 'train'), 'train', valid_classes, copy=True)
        copy_files(val_df, TRAIN_DATA_DIR, os.path.join(OUTPUT_BASE_DIR, 'val'), 'validation', valid_classes, copy=True)
        copy_files(test_df, TRAIN_DATA_DIR, os.path.join(OUTPUT_BASE_DIR, 'test'), 'test', valid_classes, copy=True)
    
    # 8. Sauvegarder les métadonnées
    save_metadata(train_df, val_df, test_df, OUTPUT_BASE_DIR)
    
    # 9. Générer un rapport
    generate_report(train_df, val_df, test_df, OUTPUT_BASE_DIR)
    
    print("\n✅ Splitting terminé avec succès!")
    print(f"📂 Données disponibles dans : {OUTPUT_BASE_DIR}/")
    print("\nStructure finale:")
    print(f"{OUTPUT_BASE_DIR}/")
    print("├── train/")
    print("│   ├── Atelectasis/")
    print("│   ├── Infiltration/")
    print("│   └── ...")
    print("├── val/")
    print("├── test/")
    print("├── train_metadata.csv")
    print("├── val_metadata.csv")
    print("├── test_metadata.csv")
    print("└── split_report.json")

if __name__ == "__main__":
    main()