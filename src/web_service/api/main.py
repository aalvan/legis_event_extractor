from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

app = FastAPI()
class PrmVideoFilePath(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: PrmVideoFilePath):
    print("transcription Start")
    transcripcionSegments,transcripcionText  = Transcripcion(request)
    print("speakers Start")
    speachers= FaceReconigtion(prmVideoFilePath)
    print("CleanText Start")
    texto=CleanText(transcripcionText,prmVideoFilePath)
    if prmModel=="spacy":
        print("SpacyNER Start")
        Entities= SpacyNER(texto,prmVideoFilePath)
    if prmModel=="beto":
        print("BetoNER Start")
        Entities= BetoNER(texto,prmVideoFilePath)

    # Cargar el archivo de video
    video = VideoFileClip(prmVideoFilePath)
    autores = getAutores(Entities)
    for autor in autores:
        autor.segments = getOficios(autor.segments)
    autoresData = getAutoresData(autores,transcripcionSegments, speachers)
    discursos = getDiscursos(autoresData)
    output_data = {"sesion": prmTitle,
                  "videoURL": "https://fake.com/"+prmVideoFilePath,
                  "texto":texto,
                  "duracioVideo" : {"start" : 0, "end" : video.duration } ,
                  "discursos" : discursos}
    with open(prmVideoFilePath +".output.json", 'w', encoding="utf-8") as newf:
        json.dump(output_data, newf, ensure_ascii=False, indent=4)

    return {"sesion": prmTitle}