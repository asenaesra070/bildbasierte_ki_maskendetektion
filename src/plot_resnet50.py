import torch                                        # PyTorch: Tensorverarbeitung, Modell-Training, GPU-Unterstützung, Autograd
import numpy as np
import matplotlib.pyplot as plt                     # Für die Visualisierung von Bildern
import seaborn as sb                                # Für Heatmaps, die Confusion Matrix wird nach der Eval geladen und als Heatmap angezeigt.

# --------------- RESNET50: Trainings und Validierung Metriken Laden --------------------------------------
metrics_resnet = torch.load("../results/resnet50_metrics_lists.pt")

# -----------------Extrahiere Listen für jede Metrik--------------------------------------------
train_loss_resnet = metrics_resnet["train_loss"]
val_loss_resnet = metrics_resnet["val_loss"]
train_acc_resnet = metrics_resnet["train_acc"]
val_acc_resnet = metrics_resnet["val_acc"]
train_prec_resnet = metrics_resnet["train_prec"]
val_prec_resnet = metrics_resnet["val_prec"]
train_rec_resnet = metrics_resnet["train_rec"]
val_rec_resnet = metrics_resnet["val_rec"]
train_f1_resnet = metrics_resnet["train_f1"]
val_f1_resnet = metrics_resnet["val_f1"]

# Erstellt die x-Achse für den Plot: Liste der Epochen von 1 bis number_epochs
epochs_resnet = list(range(1, len(train_loss_resnet) + 1))

# --------------- SimpleCNN Metrics zum Vergleich mit ResNet50 Laden --------------------------------------
metrics_simplecnn = torch.load("../results/metrics_lists.pt")
train_loss_simplecnn = metrics_simplecnn["train_loss"]
val_loss_simplecnn = metrics_simplecnn["val_loss"]
epochs_simplecnn = list(range(1, len(train_loss_simplecnn) + 1))

# ----------------- Loss Plot (Train + Validation) mit Vergleich ---------------------------------------
# ----------------- Loss Plot (Train + Validation) mit Vergleich ---------------------------------------
plt.figure(figsize=(8, 5))

# ResNet50 in Grün und Rot
plt.plot(epochs_resnet, train_loss_resnet, color='green', marker='o', label='ResNet50 Training Loss')
plt.plot(epochs_resnet, val_loss_resnet, color='red', marker='s', label='ResNet50 Validation Loss')

# SimpleCNN in Blau und Orange gleich mit was ich in plot_simple_cnn
plt.plot(epochs_simplecnn, train_loss_simplecnn, color='blue', linestyle='--', marker='o', label='SimpleCNN Training Loss')
plt.plot(epochs_simplecnn, val_loss_simplecnn, color='orange', linestyle='--', marker='s', label='SimpleCNN Validation Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss pro Epoche (ResNet50 vs. SimpleCNN)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/plot_loss_perEpoch_resnet50_vs_simplecnn.png")
plt.show()
#--------------------------------------------------------------------------------------------
# -------------------- Accuracy, Precision, Recall, F1 für Trainings--------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(epochs_resnet, train_acc_resnet, label="Train Accuracy")
plt.plot(epochs_resnet, train_prec_resnet, label="Train Precision")
plt.plot(epochs_resnet, train_rec_resnet, label="Train Recall")
plt.plot(epochs_resnet, train_f1_resnet, label="Train F1")
plt.xlabel("Epoche")
plt.ylabel("Metrik-Wert")
plt.title("Metriken pro Epoche Training - ResNet50")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/plot_metrics_train_resnet50.png")
plt.show()

# -------------------- Accuracy, Precision, Recall, F1 für Validierung -----------------------
plt.figure(figsize=(10, 6))
plt.plot(epochs_resnet, val_acc_resnet, label="Val Accuracy")
plt.plot(epochs_resnet, val_prec_resnet, label="Val Precision")
plt.plot(epochs_resnet, val_rec_resnet, label="Val Recall")
plt.plot(epochs_resnet, val_f1_resnet, label="Val F1")
plt.xlabel("Epoche")
plt.ylabel("Metrik-Wert")
plt.title("Metriken pro Epoche Validation (ResNet50)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/plot_metrics_val_resnet50.png")
plt.show()
#--------------------------------------------------------------------------------------------
# -------------------- Confusion Matrix Plot ------------------------------------------------
"""
[[TN FP]
[FN TP]]

TN: True Negatives → correctly predicted no-mask
FP: False Positives → incorrectly predicted mask
FN: False Negatives → incorrectly predicted no-mask
TP: True Positives → correctly predicted mask
"""

try:
    conf_matrix = torch.load("../results/confusion_matrix_resnet50.pt", weights_only=False)
    tn, fp, fn, tp = conf_matrix.ravel()
    # Werte geben für besseres Verständnis im Console
    print(f"True Negative (TN): {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Positive (TP): {tp}")
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - ResNet50")
    plt.colorbar()

    # Achsenbeschriftung
    class_names = ['without_mask', 'with_mask']
    plt.xticks(np.arange(len(class_names)), class_names)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Tatsächliche Klasse")

    # Werte in die Zellen schreiben
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center", color="red", fontsize=12)
    plt.tight_layout()
    plt.savefig("../results/confusion_matrix_plot_resnet50.png")
    plt.show()
except Exception as e:
    print("Es gibt keine Confusion Matrix gefunden oder Ladefehler:")
    print(e)
