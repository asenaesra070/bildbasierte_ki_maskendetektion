import torch                                                        # PyTorch: Tensorverarbeitung, Modell-Training, GPU-Unterstützung, Autograd
import torch.nn as nn                                               # Neuronale Netze und Schichten (z.B. Linear, ReLU)
import os                                                           # Automatische Ordnerverwaltung
from torch.utils.data import DataLoader, Subset                     # DataLoader für Batches + Teilmengen (Train/Test-Splits)
from torchvision import datasets, transforms                        # Laden und Transformieren von Bilddatensätzen
from sklearn.model_selection import train_test_split                # Aufteilen der Daten in Trainings- und Testsets
# Wichtige Metriken werden importiert
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model.resnet50_transfer import get_resnet50_model              # get_resnet50_model modell importiert werden
from data.data_loader import get_balanced_subset_loader_RWMFD       # DataLoader für den ausgewogenen Subset

"""Die Eval-Datei lädt nur das zuletzt trainierte Modell und liefert ein einmaliges Endtestergebnis. Eval verfolgt den Verlust nicht für jede Epoche."""

# Gerät festlegen (GPU, falls vorhanden, sonst CPU) und also GPU ist schneller als CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Mein Gerät: {device}")
# -------------------------------------------------------------------------------------------------------------------------------------------
# Datenpfad
data_dir = r"C:\Users\guler\OneDrive - Hochschule Düsseldorf\Desktop\BildBasierteKI\bildbasierte_ki_maskendetektion\data\RWMFD"
# -------------------------------------------------------------------------------------------------------------------------------------------
# Nur val_loader laden
_, val_loader, _ = get_balanced_subset_loader_RWMFD(data_dir, batch_size=64)
# -------------------------------------------------------------------------------------------------------------------------------------------
# Modell laden
model = get_resnet50_model(num_classes=2, pretrained=True, feature_extract=True).to(device)
print(f"Wo ist ResNet50 - Modell im Gerät?", next(model.parameters()).device)
# -------------------------------------------------------------------------------------------------------------------------------------------
# Model Gewichte in hier reloaded, letzte trainierte model
model.load_state_dict(torch.load("../results/resnet50_model.pth", map_location=device))
# Trainiertes Modell wird in den Evaluierungsmodus versetzt
model.eval()
# --------------------------------------- Evaluation Starten --------------------------------------------------------------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        labels = 1 - labels                                                 # Klassen umkehren, damit with_mask = 1, without_mask = 0
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
# ----------------------------------------- Metriken Berechnen-------------------------------------------------------------------

acc = accuracy_score(all_labels, all_preds)              # Berechnet den Anteil der korrekt vorhergesagten Labels (Genauigkeit)
prec = precision_score(all_labels, all_preds)            # Misst den Anteil der tatsächlichen Positiven unter den als positiv vorhergesagten (Präzision)
rec = recall_score(all_labels, all_preds)                # Misst den Anteil der korrekt erkannten Positiven an allen tatsächlichen Positiven (Sensitivität)
f1 = f1_score(all_labels, all_preds)                     # Harmonischer Mittelwert von Präzision und Recall (F1-Score)
conf_matrix = confusion_matrix(all_labels, all_preds)    # Erstellt eine Matrix mit den tatsächlichen vs. vorhergesagten Klassen (Konfusionsmatrix)
# -------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------  Ausgabe Für Evaluierte Metrik ------------------------------------------------------------------
print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall: {rec*100:.2f}%")
print(f"F1-Score: {f1*100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)

# ------------------ Plotdatei für Confusion Matrix wird dann automatisch von dort übernommen-----------------
torch.save(conf_matrix, "../results/confusion_matrix_resnet50.pt")
print(f"Confusion Matrix wurde gespeichert : ../results/confusion_matrix_resnet50.pt")
