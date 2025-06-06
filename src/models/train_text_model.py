# src/models/train_text_model.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd

class ClinicalNoteDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_clinicalbert(csv_path, model_repo_id, output_dir, epochs=4, batch_size=16, lr=2e-5, device=None):
    """
    Entrena ClinicalBERT para clasificación de tratamiento.
    csv_path: archivo que contiene columnas 'input_text' y 'label' (numérico).
    model_repo_id: repo id en HF (por ejemplo "emilyalsentzer/Bio_ClinicalBERT").
    output_dir: carpeta donde guardar vocab, config y pesos.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar datos
    df = pd.read_csv(csv_path)
    X = df['input_text'].tolist()
    y = df['label'].tolist()

    # 2. LabelEncoder
    le = LabelEncoder()
    df['label'] = le.fit_transform(y)
    # Guardar mapping en config si vas a subir a HF (más adelante)

    # 3. Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # 4. Tokenizer y datasets
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
    train_dataset = ClinicalNoteDataset(X_train, y_train, tokenizer, max_len=50)
    val_dataset = ClinicalNoteDataset(X_val, y_val, tokenizer, max_len=50)
    test_dataset = ClinicalNoteDataset(X_test, y_test, tokenizer, max_len=50)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 5. Modelo
    model = AutoModelForSequenceClassification.from_pretrained(model_repo_id, num_labels=len(le.classes_))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_training_steps = len(train_loader) * epochs
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # 6. Entrenamiento
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct += (preds == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)

        # Validación
        model.eval()
        val_loss, correct_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                val_loss += loss.item()
                correct_val += (preds == labels).sum().item()

        val_acc = correct_val / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print("🌟 Mejor modelo guardado en", output_dir)

    print("Entrenamiento de ClinicalBERT finalizado. Best Val Acc:", best_val_acc)

    return model, test_loader, le


if __name__ == "__main__":
    # Ejemplo de uso:
    csv_path = "..\Proy_Biociencias\Proyecto_Biociencias\src\data\clinical_notes_labeled.csv"
    model_repo_id = "emilyalsentzer/Bio_ClinicalBERT"
    output_dir = "checkpoints/clinicalbert_best"
    train_clinicalbert(csv_path, model_repo_id, output_dir)
