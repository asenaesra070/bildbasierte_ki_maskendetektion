import torch                                                        # PyTorch: Tensor verarbeitung, Modell-Training, GPU-Unterstützung, Autograd
import numpy as np
import torch.nn as nn                                               # Neuronale Netze und Schichten (z.B. Linear, ReLU)
import torch.optim as optim                                         # Optimierer für Gewichtsanpassung
import os                                                           # Automatische Ordnerverwaltung
import gc
#------------------------------------------------------------------------------------------------------------------------------------------------------------
from model.resnet50_transfer import get_resnet50_model              # Funktion um ResNet50 mit Transfer Learning zu laden
from data.data_loader import get_balanced_subset_loader_RWMFD       # DataLoader für den ausgewogenen Subset
# Wichtige Metriken werden importiert
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#------------------------------------------------------------------------------------------------------------------------------------------------------------
# Gerät festlegen (GPU, falls vorhanden, sonst CPU) und also GPU ist schneller als CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Mein Gerät: {device}")
# ---------------------------------------------------------------------------------
# Modell initialisieren und das Modell an das Gerät senden werden. In hier nur letzte Schicht wird trainiert!
model = get_resnet50_model(num_classes=2, pretrained=True, feature_extract=True).to(device)
print(f"Wo ist ResNet50 - Modell im Gerät?", next(model.parameters()).device)
# ------------------- Hyperparameters -------------------------------------------
learning_rate = 0.001
batch_size = 64
# Bestimmt, wie oft das Modell den gesamten Trainingsdatensatz von Anfang bis Ende durchsucht. 10,20,100 … aber so viel macht overfitting
number_epochs = 10
# Verlustfunktion (keine Gewichte explizit, da ResNet oft schon gut mit Balanced Subset arbeitet)
class_weights = torch.tensor([0.7, 0.3]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# ---------------------------------------------------------------------------------
# Datenpfad
data_dir = r"C:\Users\guler\OneDrive - Hochschule Düsseldorf\Desktop\BildBasierteKI\bildbasierte_ki_maskendetektion\data\RWMFD"
# ---------------------------------------------------------------------------------

# Lädt Bilder aus dem RMFD-Datensatz mit Image Etiketten ('with_mask': 1, 'without_mask': 0) und einen DataLoader mit Subset : 2560 Images
# train_loader, _, class_dict = get_data_loader_RWMFD(data_dir, batch_size = batch_size)
train_loader, val_loader, class_dict = get_balanced_subset_loader_RWMFD(data_dir, batch_size=batch_size)
#----- Es wurde die Listen erstellt für Trainings, um die Ausgaben aller Metriken in der plot.py darzustellen.

train_loss_list = []
train_acc_list = []
train_prec_list = []
train_rec_list = []
train_f1s_list = []

#----- Es wurde die Listen erstellt für Validierung (Test Datensatz) , um die Ausgaben aller Metriken in der plot.py darzustellen.

val_loss_list = []
val_acc_list = []
val_prec_list = []
val_rec_list = []
val_f1s_list = []
#-------------------------- Trainingsschleife ---------------------------------------
print("Training started für ResNet50 Transfer Modell...")
for epoch in range(number_epochs):
    # Modell in Trainingsmodus setzen
    model.train()
    epoch_loss = 0.0
    all_preds, all_labels  = [], []
    for images, labels in train_loader:
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

    # Ergebnisse in Listen speichern für z.B. Metriken wie Genauigkeit, Präzision, Rückruf und F1-Score.
    train_loss_list.append(epoch_loss)
    train_acc_list.append(accuracy)
    train_prec_list.append(precision)
    train_rec_list.append(recall)
    train_f1s_list.append(F1)

    print(f"Epoch {epoch+1}/{number_epochs} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {F1:.4f}")
# ------------------------------- Validation-Phase Schleife ------------------------------------------------
    model.eval()
    val_epoch_loss = 0.0
    val_all_preds = []
    val_all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            labels = 1 - labels
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_epoch_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_all_preds.extend(preds.cpu().numpy())
            val_all_labels.extend(labels.cpu().numpy())

    # Validierungsmetriken berechnen
    val_accuracy = accuracy_score(val_all_labels, val_all_preds)
    val_precision = precision_score(val_all_labels, val_all_preds)
    val_recall = recall_score(val_all_labels, val_all_preds)
    val_F1 = f1_score(val_all_labels, val_all_preds)

    # Ergebnisse in Validierungslisten speichern
    val_loss_list.append(val_epoch_loss)
    val_acc_list.append(val_accuracy)
    val_prec_list.append(val_precision)
    val_rec_list.append(val_recall)
    val_f1s_list.append(val_F1)

    print(f"Validation - Loss: {val_epoch_loss:.4f} | Accuracy: {val_accuracy:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_F1:.4f}")
    model.train()            # Setzt das Modell zurück in den Trainingsmodus für die nächste Epoche
    gc.collect()             # Speicherbereinigung nach jeder Epoche (gegen Speicherüberlastung)

# ------------------- Modell Speichern -----------------------------
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "../results/resnet50_model.pth")
print("ResNet50 Modell gespeichert: results/resnet50_model.pth")

metrics = {
    'train_loss': train_loss_list,
    'train_acc': train_acc_list,
    'train_prec': train_prec_list,
    'train_rec': train_rec_list,
    'train_f1': train_f1s_list,
    'val_loss': val_loss_list,
    'val_acc': val_acc_list,
    'val_prec': val_prec_list,
    'val_rec': val_rec_list,
    'val_f1': val_f1s_list
}
# Mittelwert, der durchschnittliche Erfolg des Trainingsprozesses.
print("Durchschnittliche Training Accuracy:", sum(train_acc_list) / len(train_acc_list))
# Mittelwert, der durchschnittliche Erfolg des Evaluirensprozesses.
print("Durchschnittliche Validation Accuracy:", sum(val_acc_list) / len(val_acc_list))

torch.save(metrics, "../results/resnet50_metrics_lists.pt")
print("Trainings und Validierungsmesswerte gespeichert: results/resnet50_metrics_lists.pt")