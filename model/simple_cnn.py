import torch
import torch.nn as nn
import torch.nn.functional as Fu

# Ein einfaches CNN-Modell zur binären Klassifikation (Maske vs. keine Maske)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Erstes Convolutional Layer: Eingang hat 3 Kanäle (RGB), Ausgang hat 128 Feature Maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1)
        # [3, 224, 224], die Pixelanzahl ist bei Transform Resize auf 224 festgelegt.

        # Zweites Convolutional Layer: nimmt 128 Kanäle, gibt 64 aus
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        # [128, 224, 224] → [64, 224, 224]

        # MaxPooling reduziert die räumliche Dimension um die Hälfte (2x2 Fenster)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 224 → 112 → 56

        # Fully connected Layer: Eingang ist das geflattete Feature-Map (64*56*56), Ausgang ist 128 Neuronen
        # Schicht fc1 nimmt 200704 Zahlen auf → verbindet sie mit 128 Neuronen.
        self.fc1 = nn.Linear(64 * 56 * 56, 128)


        # Letzte Schicht: 128 Neuronen gibt als Ausgabe ist 2 Klassen (Maske / Keine Maske)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # Erste Convolution + ReLU + Pooling
        x = self.pool(Fu.relu(self.conv1(x)))  # Ausgabe: [128, 112, 112]

        # Zweite Convolution + ReLU + Pooling
        x = self.pool(Fu.relu(self.conv2(x)))  # Ausgabe: [64, 56, 56]

        # Umwandlung in einen flachen Vektor (Flatten)
        x = x.view(-1, 64 * 56 * 56)  # Ausgabe: [Batchgröße, 200704]
        # -1 bedeutet passt die Anzahl der Bilder an
        # Erstes Fully Connected Layer + ReLU
        x = Fu.relu(self.fc1(x))  # Ausgabe: [Batchgröße, 128]

        # Letzte Fully Connected Layer → Ausgabe: 2 Klassen
        x = self.fc2(x)  # Ausgabe: [Batchgröße, 2]

        return x

torch.save(model.state_dict(), "...results/simple_cnn_model.pth")
# ?ZEILE 44 Bei diesen Zahlen handelt es sich normalerweise um Logits, die dann mit Softmax wie folgt interpretiert werden: „Wie hoch ist die Wahrscheinlichkeit, dass dieses Bild maskiert ist?“