# src/api/app.py

import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.api.crop_utils import CropBlackBorders

app = FastAPI(title="API: Tumor & Tratamiento (Paso a paso)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_model():
    repo_id = "jherronr/efficientnet-brain-tumor-classifier"
    filename = "pytorch_model.bin"
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    model = models.efficientnet_b0(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 3)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

image_model = load_image_model()

image_transform = transforms.Compose([
    CropBlackBorders(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

idx2label_image = {
    0: "brain_glioma",
    1: "brain_menin",
    2: "brain_tumor"
}


def load_text_model():
    repo_id = "jherronr/clinicalbert-treatment-classifier"
    text_model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    text_model.to(device)
    text_model.eval()
    text_tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return text_model, text_tokenizer

text_model, text_tokenizer = load_text_model()

# Etiquetas de tratamiento corregidas manualmente
idx2label_text = {
    0: "Surgery",
    1: "chemotherapy",
    2: "radiation therapy",
    3: "close monitoring"
}


def build_input_text(condition: str, clinical_note: str) -> str:
    return f"Condition: {condition}. Clinical note: {clinical_note}"


@app.post("/predict_image")
async def predict_image(image_file: UploadFile = File(...)):
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="Formato no soportado. Use JPG o PNG.")

    try:
        contents = await image_file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al leer la imagen: {e}")

    try:
        img_cropped = image_transform(pil_image)
        img_tensor = img_cropped.unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en preprocesamiento: {e}")

    try:
        with torch.no_grad():
            outputs_img = image_model(img_tensor)
            _, preds_img = torch.max(outputs_img, dim=1)
            tumor_label = idx2label_image[preds_img.item()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia: {e}")

    return JSONResponse(content={"tumor_prediction": tumor_label})


@app.post("/predict_treatment")
async def predict_treatment(
    condition: str = Form(...),
    clinical_note: str = Form(...)
):
    if not condition.strip():
        raise HTTPException(status_code=400, detail="El campo 'condition' no puede estar vacío.")
    if not clinical_note.strip():
        raise HTTPException(status_code=400, detail="El campo 'clinical_note' no puede estar vacío.")

    input_text = build_input_text(condition, clinical_note)

    try:
        encoding = text_tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=50,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en tokenización: {e}")

    try:
        with torch.no_grad():
            outputs_txt = text_model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds_txt = torch.max(outputs_txt.logits, dim=1)
            treatment_label = idx2label_text[preds_txt.item()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia de texto: {e}")

    return JSONResponse(content={"treatment_prediction": treatment_label})
