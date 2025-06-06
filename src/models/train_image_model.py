# src/models/train_image_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets
from torch.utils.data import DataLoader, random_split
from src.data.preprocessing_images import get_image_transform

def train_efficientnet(dataset_path, output_checkpoint, epochs=10, batch_size=32, lr=0.001, device=None):
    """
    Entrena EfficientNet-B0 (con transferencia de aprendizaje) para clasificar tumor cerebral en 3 clases.
    dataset_path: ruta a carpeta con subcarpetas por clase (ImageFolder).
    output_checkpoint: ruta para guardar pesos.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Preprocesamiento
    transform = get_image_transform(resize_to=(224, 224))
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # 2. Split train/val/test
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 3. Cargar preentrenado y modificar capa final
    model = models.efficientnet_b0(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)

        # Validación
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += (preds == labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_epoch_acc:.4f}")

        # Guardar si mejora
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), output_checkpoint)
            print("🌟 Mejor modelo guardado.")

    print("Entrenamiento de EfficientNet finalizado. Best Val Acc:", best_val_acc)
    return model, test_loader


if __name__ == "__main__":
    # Ejemplo de uso:
    # python src/models/train_image_model.py
    dataset_path = "..\Proy_Biociencias\Proyecto_Biociencias\Brain Cancer - MRI dataset\Brain_Cancer raw MRI data\Brain_Cancer"
    output_checkpoint = "checkpoints/efficientnet_best.pth"
    model, test_loader = train_efficientnet(dataset_path, output_checkpoint)
