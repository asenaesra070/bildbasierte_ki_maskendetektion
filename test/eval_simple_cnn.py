import torch                                                        # PyTorch: Tensorverarbeitung, Modell-Training, GPU-Unterstützung, Autograd
import torch.nn as nn                                               # Neuronale Netze und Schichten (z.B. Linear, ReLU)

from torch.utils.data import DataLoader, Subset                     # DataLoader für Batches + Teilmengen (Train/Test-Splits)
from torchvision import datasets, transforms                         # Laden und Transformieren von Bilddatensätzen
from sklearn.model_selection import train_test_split                 # Aufteilen der Daten in Trainings- und Testsets

# Wichtige Metriken werden importiert
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model.simple_cnn import SimpleCNN                              # SimpleCNN modell importiert werden
from data.data_loader import get_data_loader_RWMFD                  # Angepasste DataLoader-Funktion (enthält val_loader)
from data.data_loader import get_balanced_subset_loader_RWMFD       # Angepasste DataLoader-Funktion (enthält val_loader) mit Subset
# Gerät festlegen (GPU, falls vorhanden, sonst CPU) und also GPU ist schneller als CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Mein Gerät: {device}")

# Datenpfad
data_dir = r"C:\Users\guler\OneDrive - Hochschule Düsseldorf\Desktop\BildBasierteKI\bildbasierte_ki_maskendetektion\data\RWMFD"

# Nur val_loader laden (Trainloader brauchen wir hier nicht)
# _, val_loader, class_dict = get_data_loader_RWMFD(data_dir, batch_size=64)
val_loader, class_dict = get_balanced_subset_loader_RWMFD(data_dir, batch_size=64)
# Modell laden
model = SimpleCNN().to(device)
print(f"Wo ist SimpleCNN im Gerät? ", next(model.parameters()).device)
model.load_state_dict(torch.load("../results/simple_cnn_model.pth", map_location=device))
model.eval()

# Evaluation starten
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        labels = 1 - labels  # Klassen umkehren, damit with_mask = 1, without_mask = 0
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metriken berechnen
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

# Ausgabe
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)