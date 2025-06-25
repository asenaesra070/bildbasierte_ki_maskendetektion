import torch                                                        # PyTorch: Tensor verarbeitung, Modell-Training, GPU-Unterstützung, Autograd
import numpy as np
import torch.nn as nn                                               # Neuronale Netze und Schichten (z.B. Linear, ReLU)
import torch.optim as optim                                         # Optimierer für Gewichtsanpassung
from torchvision.utils import save_image                            # Visualisierung der Ergebnisse
from model.simple_cnn import SimpleCNN                              # SimpleCNN modell importiert werden
from data.data_loader import get_data_loader_RWMFD                  # get_data_loader_RWMFD daten importiert werden
# Wichtige Metriken werden importiert
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os                                                           # Automatische Ordnerverwaltung
import gc
from data.data_loader import get_balanced_subset_loader_RWMFD

# Gerät festlegen (GPU, falls vorhanden, sonst CPU) und also GPU ist schneller als CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Mein Gerät: {device}")

# Modell initialisieren und das Modell an das Gerät senden werden
model = SimpleCNN().to(device)
print(f"Wo ist SimpleCNN im Gerät?", next(model.parameters()).device)

# ------------------- Hyperparameters -------------------------------------------
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Bestimmt, wie oft das Modell den gesamten Trainingsdatensatz von Anfang bis Ende durchsucht. 10,20,100 … aber so viel macht overfitting
number_epochs = 4

# Datenpfad und BatchSize
data_dir = r"C:\Users\guler\OneDrive - Hochschule Düsseldorf\Desktop\BildBasierteKI\bildbasierte_ki_maskendetektion\data\RWMFD"
batch_size = 64
# Lädt Bilder aus dem RMFD-Datensatz mit Image Etiketten ('with_mask': 1, 'without_mask': 0) und einen DataLoader
# train_loader, _, class_dict = get_data_loader_RWMFD(data_dir, batch_size = batch_size)
train_loader, class_dict = get_balanced_subset_loader_RWMFD(data_dir, batch_size=batch_size)
# Modell in Trainingsmodus setzen
model.train()
#-------------------------- Trainingsschleife ---------------------------------------
print("Training started...")
for epoch in range(number_epochs):
    epoch_loss = 0.0
    # Die Listen um Trainings Metrics zu halten
    # am Ende jeder Epoche berechnen wir Metriken wie Genauigkeit, Präzision, Rückruf und F1-Score.
    all_preds = []
    all_labels = []
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx +1} geladen ...")
        if batch_idx >= 10:
            print("Nur die ersten 10 Batches werden zum Testen verwendet.")
            break
        # Labels umkehren, weil im DataLoader 'with_mask': 1, wir aber 'without_mask': 0 erwarten. Weil ImageFolder Ordner alphabetisch liest
        labels = 1 - labels

        # images und label an das Mein Gerät senden werden
        images, labels = images.to(device), labels.to(device)

        # Gradienten zurücksetzen aus vorherigem Batch
        optimizer.zero_grad()

        # Vorwärtsdurchlauf: Das Modell generiert aus den Bildern Rohklassenwerte, sogenannte Logits.
        outputs = model(images)

        # Verlust berechnen: Die tatsächliche Bezeichnung wird mit der Modellausgabe verglichen.
        loss = criterion(outputs, labels)
        # Verlustwert pro Batch als Float und addiert ihn zum Gesamtverlust.
        epoch_loss += loss.item()

        # Rückwärtsdurchlauf (berechnen Gradient) + Optimierung
        loss.backward()
        # Aktualisiert die Gewichte des Modells basierend auf Gradienten
        optimizer.step()

        # Vorhersagen speichern
        # Hint : values, indices = torch.max(tensor, dim)
        _, preds = torch.max(outputs, 1)
        # Aber sklearn.metrics-Funktionen (accuracy_score, precision_score usw.) erfordern NumPy-Arrays
        all_preds.extend(preds.cpu().numpy())
        # all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

    # Metriken berechnen
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    F1 = f1_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}/{number_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {F1:.4f}")
    gc.collect()            # Speicherbereinigung nach jeder Epoche (gegen Speicherüberlastung)
# Modell speichern
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/simple_cnn_model.pth")
print("Modell gespeichert: results/simple_cnn_model.pth")