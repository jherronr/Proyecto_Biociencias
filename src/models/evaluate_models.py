# src/models/evaluate_models.py

import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_image_model(model, test_loader, idx2label):
    """
    Evalúa el modelo de imagen en el test set y muestra métricas.
    """
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    print("=== Classification Report (Imagen) ===")
    print(classification_report(y_true, y_pred, target_names=list(idx2label.values())))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(idx2label.values()),
                yticklabels=list(idx2label.values()))
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión - Modelo de Imagen")
    plt.show()


def evaluate_text_model(model, test_loader, idx2label):
    """
    Evalúa el modelo de texto en el test set y muestra métricas.
    """
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    print("=== Classification Report (Texto) ===")
    print(classification_report(y_true, y_pred, target_names=list(idx2label.values())))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=list(idx2label.values()),
                yticklabels=list(idx2label.values()))
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión - Modelo de Texto")
    plt.show()
