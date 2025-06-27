import torch                                        # PyTorch: Tensorverarbeitung, Modell-Training, GPU-Unterstützung, Autograd
import numpy as np
import matplotlib.pyplot as plt                     # Für die Visualisierung von Bildern
import seaborn as sb                                # Für Heatmaps, die Confusion Matrix wird nach der Eval geladen und als Heatmap angezeigt.

# --------------- Trainings und Validierung Metriken Laden --------------------------------------
metrics = torch.load("../results/metrics_lists.pt")

# -----------------Extrahiere Listen für jede Metrik--------------------------------------------
train_loss = metrics["train_loss"]
val_loss = metrics["val_loss"]
train_acc = metrics["train_acc"]
val_acc = metrics["val_acc"]
train_prec = metrics["train_prec"]
val_prec = metrics["val_prec"]
train_rec = metrics["train_rec"]
val_rec = metrics["val_rec"]
train_f1 = metrics["train_f1"]
val_f1 = metrics["val_f1"]

# Erstellt die x-Achse für den Plot: Liste der Epochen von 1 bis number_epochs
epochs = list(range(1, len(train_loss) + 1))

# ----------------- Loss Plot (Train + Validation) ---------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, metrics["train_loss"], marker='o', label='Training Loss')
plt.plot(epochs, metrics["val_loss"], marker='s', label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss pro Epoche (Training und Validation)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/plot_loss_perEpoch_train_val.png")
plt.show()
#--------------------------------------------------------------------------------------------
# -------------------- Accuracy, Precision, Recall, F1 für Trainings--------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics["train_acc"], label="Train Accuracy")
plt.plot(epochs, metrics["train_prec"], label="Train Precision")
plt.plot(epochs, metrics["train_rec"], label="Train Recall")
plt.plot(epochs, metrics["train_f1"], label="Train F1")
plt.xlabel("Epoche")
plt.ylabel("Metrik-Wert")
plt.title("Metriken pro Epoche Training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/plot_metrics_train.png")
plt.show()
# -------------------- Accuracy, Precision, Recall, F1 für Validierung -----------------------
plt.figure(figsize=(10, 6))
plt.plot(epochs, metrics["val_acc"], label="Val Accuracy")
plt.plot(epochs, metrics["val_prec"], label="Val Precision")
plt.plot(epochs, metrics["val_rec"], label="Val Recall")
plt.plot(epochs, metrics["val_f1"], label="Val F1")
plt.xlabel("Epoche")
plt.ylabel("Metrik-Wert")
plt.title("Metriken pro Epoche Validation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/plot_metrics_val.png")
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
    conf_matrix = torch.load("../results/confusion_matrix.pt", weights_only=False)
    tn, fp, fn, tp = conf_matrix.ravel()
    # Werte geben für besseres Verständnis im Console
    print(f"True Negative (TN): {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Positive (TP): {tp}")
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
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
    plt.savefig("../results/confusion_matrix_plot.png")
    plt.show()
except Exception as e:
    print("E gibt keine Confusion Matrix gefunden oder Ladefehler:")
    print(e)

