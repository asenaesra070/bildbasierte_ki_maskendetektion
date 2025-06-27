import torch                            # PyTorch: Tensor verarbeitung, Modell-Training, GPU-Unterstützung, Autograd
import torch.nn as nn                   # Neuronale Netze und Schichten (z.B. Linear, ReLU)
from torchvision import models          # Vorgefertigte Modelle (ResNet50, die letzte Schicht der 50 geladenen Schichten für die binäre Klassifikation geeignet machen)



#-------------------------------------------- Function um vortranierte Modelle (ResNet50) zu laden -----------------------------------------------
def get_resnet50_model(num_classes=2, pretrained=True, feature_extract=True):

    # Lädt ein ResNet50-Modell und passt die letzte Schicht an. In hier , Modell mit vorgelernten Gewichten laden
    model = models.resnet50(pretrained=True)

    if feature_extract:

        # Friert alle Gewichte ein, damit nur der Feature Extraction trainiert wird
        for param in model.parameters():
            param.requires_grad = False

    # Es nimmt automatisch die Eingabegröße der letzten Ebene
    num_ftrs = model.fc.in_features
    # setzt seine Ausgabe auf 2, with_mask und without_mask
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model