import torch                                                    # PyTorch: Tensor verarbeitung, Modell-Training, GPU-Unterstützung, Autograd
import matplotlib.pyplot as plt                                 # Für die Visualisierung von Bildern
from torchvision.utils import make_grid, save_image             # Hilft beim Anzeigen mehrerer Bilder in einem Raster
from data.data_loader import get_balanced_subset_loader_RWMFD   # Lädt den ausgewogenen Subset-Loader
from model.resnet50_transfer import get_resnet50_model          # get_resnet50_model modell importiert werden
import os                                                       # Automatische Ordnerverwaltung


# Gerät festlegen (GPU, falls vorhanden, sonst CPU) und also GPU ist schneller als CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# -------------------------------------------------------------------------------------------------
# Modell initialisieren und Gewichte laden
model = get_resnet50_model(num_classes=2, pretrained=True, feature_extract=True).to(device)
model.load_state_dict(torch.load("../results/resnet50_model.pth", map_location=device))
model.eval()                                                   # Setzt das Modell in den Evaluierungsmodus
# -------------------------------------------------------------------------------------------------
# Datensatzpfad
data_dir = r"C:\Users\guler\OneDrive - Hochschule Düsseldorf\Desktop\BildBasierteKI\bildbasierte_ki_maskendetektion\data\RWMFD"

# DataLoader für den ausgewogenen Subset laden
_, val_loader, class_dict = get_balanced_subset_loader_RWMFD(data_dir, batch_size=64)

# Einen Batch aus dem Validierungs-Set entnehmen
images, labels = next(iter(val_loader))

# Nur die ersten 20 Bilder verwenden
images, labels = images[:20], labels[:20]

# Klassenlabels umkehren, damit 'with_mask' → 1 und 'without_mask' → 0
labels = 1 - labels

# Modellvorhersage ohne Gradienten berechnung
with torch.no_grad():
    outputs = model(images.to(device))
    _, preds = torch.max(outputs, 1)  # Klassenvorhersage: 0 oder 1

# Bilder in CPU umwandeln zur Visualisierung und zum Speichern
images = images.cpu()

# Matplotlib: Bilder in einem Raster anzeigen
plt.figure(figsize=(15, 6))
grid_img = make_grid(images, nrow=5)
plt.imshow(grid_img.permute(1, 2, 0))  # Kanäle (C,H,W) → (H,W,C)

# Vorhersage pro Bild anzeigen – jedes Bild bekommt seine eigene Beschriftung
fig, axs = plt.subplots(3, 5, figsize=(15, 6))  # 3 Zeilen, 5 Spalten (für 15 Bilder)
axs = axs.flatten()  # 2D-Array in 1D umwandeln für einfachen Zugriff

for idx in range(15):
    img = images[idx].permute(1, 2, 0).numpy()  # Tensor -> NumPy für plt.imshow
    axs[idx].imshow(img)  # Bild anzeigen
    axs[idx].set_title("with_mask" if preds[idx] == 1 else "without_mask", fontsize=9)  # Titel (Vorhersage)
    axs[idx].axis("off")  # Achsen ausblenden

plt.tight_layout()
plt.show()

# ----- Bilder in einem lokalen Ordner speichern -----
# Speicherordner festlegen
save_dir = "predicted_samples_resnet50"
os.makedirs(save_dir, exist_ok=True)  # Ordner erstellen, falls nicht vorhanden

# Bilder speichern mit einfachen Namen (z.B. sample_1_resnet50.png)
for idx, img in enumerate(images):
    save_path = os.path.join(save_dir, f"sample_{idx+1}_resnet50.png")
    save_image(img, save_path)
