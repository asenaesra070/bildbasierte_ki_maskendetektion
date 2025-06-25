import torch
# Importiert Module für Bilddatensätze und Transformationen
from torchvision import datasets, transforms

# Ermöglicht das Laden von Daten in Batches
from torch.utils.data import DataLoader, Subset

# Teilen Sie Arrays oder Matrizen in zufällige Trainings- und Test-Teilmengen auf
from sklearn.model_selection import train_test_split


# DataLoader → data_dir : pfad von wo stehen alle Bildern →  dataset
# transform → wandelt Bildern zur Tensoren und skaliert bestimmte Pixel (z.B. 128x128)

"""
Ein DataLoader wird für Bilder im folgenden Format erstellt
data_dir/
    ├── with_mask/
    └── without_mask/
"""

def get_data_loader_RWMFD(data_dir, batch_size = 64, val_ratio=0.2):
    # Definiert Transformationen für die Bilder: Größe anpassen und in Tensor konvertieren
    transform = transforms.Compose([
        # Diese Zeile erzwingt, dass jedes Bild 224 x 224 groß ist, sodass das Modell mit Tensoren fester Größe arbeiten kann
        # Es erhöht die Prozessorlast (GPU/CPU) ein bisschen aber und ist für CNN und RestNet50 Vergleichung in data geeignet.
        transforms.Resize((224, 224)),
        # Wandelt Bildern in Tensoren um: (0.0 - 1.0)
        transforms.ToTensor(),
    ])
    # Liest Bilder mit Klassenstruktur da unter data/ aus dem Verzeichnis mithilfe von ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    # Erstellt den DataLoader mit definierter Batch-Größe und zufälliger Durchmischung
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    class_dict = dataset.class_to_idx           #  weist jedem Ordnernamen einen Index zu.
    # Aufteilung in Trainings(%80) -und Validierungsdaten
    total_size = len(dataset)
    indices = list(range(total_size))
    # um Training zu prüfen, als %80 Training und %20 Testbildern teilen werden.
    # stratify → sorgt dafür, dass Klassenverhältnis in train/test ähnlich bleibt
    train_indices, val_indices = train_test_split(indices, test_size=val_ratio, stratify=[dataset.targets[i] for i in indices])

    #Trainings Datasets
    train_dataset = Subset(dataset, train_indices)
    # Validierung Datasets
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_dict
def get_balanced_subset_loader_RWMFD(data_dir, batch_size=64, subset_size_per_class=1280):
    """
    Erstellt einen ausgewogenen (balanced) Subset-Dataloader aus dem RWMFD-Datensatz.
    Nimmt z.B. 1280 Bilder aus jeder Klasse ('with_mask', 'without_mask') → insgesamt 2560 Bilder.
    Das beschleunigt das Training & sichert faire Klassifikation.
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_to_idx = dataset.class_to_idx

    # Labels extrahieren
    targets = [sample[1] for sample in dataset.samples]

    # Für jede Klasse die passenden Indizes finden
    class_indices = {cls: [] for cls in set(targets)}
    for idx, label in enumerate(targets):
        if len(class_indices[label]) < subset_size_per_class:
            class_indices[label].append(idx)

    # Subset zusammenstellen
    selected_indices = class_indices[0] + class_indices[1]
    subset = Subset(dataset, selected_indices)

    # DataLoader erzeugen
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    return loader, class_to_idx

# --- Test (nur falls als Skript ausgeführt) ---
#jede Zahl in diesem Tensor stellt die Klassenbezeichnung eines Fotos dar. ImagesLabel : 0 oder 1
if __name__ == "__main__":
    train_loader, val_loader, class_dict = get_data_loader_RWMFD(
        r"C:\Users\guler\OneDrive - Hochschule Düsseldorf\Desktop\BildBasierteKI\bildbasierte_ki_maskendetektion\data\RWMFD",
        batch_size=64
    )
    print("Train batch size:", len(next(iter(train_loader))[0]))
    print("Validation batch size:", len(next(iter(val_loader))[0]))
    # Gibt die Länge aller Instanzen zurück, da sie an das Subset-Objekt gebunden sind.
    print("Train dataset size:", len(train_loader.dataset))
    print("Validation dataset size:", len(val_loader.dataset))

    # Richtige Label-Zuweisung: with_mask → 1, without_mask → 0
    true_class_dict = {k: 1 - v for k, v in class_dict.items()}
    print("Korrigierter class_dict:", true_class_dict)
    # um besser zu verstehen, es ist notwendig, die Etiketten in der ersten Charge umzukehren
    for images, labels in val_loader:
        labels = 1 - labels                          # 0 -> 1, 1 -> 0
        print("Label Batch (after flip):", labels)  # Erwartet: {'with_mask': 0, 'without_mask': 1}
        break