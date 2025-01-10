from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

id2label = {
    0: 'B-AUTOR',
    1: 'B-DESTINO',
    2: 'B-EVENTO',
    3: 'B-MATERIA',
    4: 'I-AUTOR',
    5: 'I-DESTINO',
    6: 'I-EVENTO',
    7: 'I-MATERIA',
    8: 'O'
}
label2id = {v: k for k, v in id2label.items()}
MODEL_PATH = '/workspaces/legis_event_extractor/models/ner_output/checkpoint-90'
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

model.config.id2label = id2label
model.config.label2id = label2id

app = FastAPI()
class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_labels = logits.argmax(dim=-1)
    predicted_labels_text = [id2label[label] for label in predicted_labels[0]]

    return {"predictions": predicted_labels_text}