import whisper
import copy
import spacy
import glob
import json
from pathlib import Path
import cv2
import insightface
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import math
from numpy.linalg import norm
import insightface
from pathlib import Path
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from transformers import  AutoTokenizer,AutoModelForTokenClassification
from transformers import pipeline
from moviepy import VideoFileClip
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from web_service.api.main import Segmento

def Transcripcion(prmVideoFilePath):
    model = whisper.load_model("turbo")
    result = model.transcribe(prmVideoFilePath, language="es" )

    output_data = {
        "language": result["language"],
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
                }
        for segment in result["segments"]
        ]
        }
    with open(prmVideoFilePath+".transcription.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    texto = result["text"].replace(". ", ". \n")
    with open(prmVideoFilePath+".transcription.txt", 'w') as f:
        f.write(texto)
    return output_data, texto


def LoadFotoEmbedding():
    files= glob.glob("data/photo_embeddings/*.npy")
    df = pd.DataFrame(columns=['Id', 'embedding'])
    dataSet = []
    for file in files:
        vId =  Path(file).stem.replace(".emb","")
        face = np.load(file)
        dataSet.append({'id':vId,'embedding':face})
    df = pd.DataFrame(dataSet)
    return df


def FaceReconigtion(prmVideoFilePath):
    app = insightface.app.FaceAnalysis()
    app.prepare(ctx_id=0)

    video = cv2.VideoCapture(prmVideoFilePath)
    frameRate = video.get(cv2.CAP_PROP_FPS)#frame rate  
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count // frameRate

    dfFotoEmbedding = LoadFotoEmbedding()
    DiputadosEmbeddings = np.array(dfFotoEmbedding['embedding'].tolist())
    DiputadosIds = dfFotoEmbedding['id'].tolist()
    speachers = []
    DipId =0

    while(video.isOpened()):
        frameId = video.get(1) #current frame number
        ret, frame = video.read()
        if (ret != True):
            break

        if (frameId % math.floor(frameRate) == 0): 
            current_time_ms = video.get(cv2.CAP_PROP_POS_MSEC)
            secs = int(current_time_ms // 1000)

            percent = (secs / duration) * 100
            bar = f"[{'#' * int(percent // 2)}{'-' * (int(100 - percent) // 2)}]"

            print(f"\r{bar} {percent:.2f}%", str(secs),"/",duration, end='\r')

            faces = app.get(frame)

            for face in faces:
                box = face.bbox.astype(int)
                area = (box[2]- box[0]) * (box[3] -  box[1])
                if area > 20000:
                    embedding = face.embedding
                    # Establecer un umbral (ajusta según tus necesidades)
                    threshold = 0.6

                    # Calcular similitudes del coseno
                    similarities = cosine_similarity(
                        embedding.reshape(1, -1),  # Embedding detectado
                        DiputadosEmbeddings                # Embeddings conocidos
                    )
                    max_similarity = similarities[0].max()
                    identified_index = similarities[0].argmax()
                    identified_person = DiputadosIds[identified_index]
                    current_time_ms = video.get(cv2.CAP_PROP_POS_MSEC)
                    # Convertir el tiempo a minutos y segundos
                    total_seconds = int(current_time_ms // 1000)
                    minutes = total_seconds // 60
                    seconds = total_seconds % 60
                    if max_similarity > threshold:
                        DipId = identified_person
                    else:
                        DipId =-1
                    speachers.append({"diputadoId":DipId, "start":total_seconds, "similarity":str(max_similarity) , "closestPerson":identified_person})
    output_data = {"speachers":speachers}
    with open(prmVideoFilePath+".speachers.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
    video.release()
    return speachers


def CleanText(prmTranscripcion, prmVideoFilePath):
    modelo = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
    stopParagraphsEmbeddings = np.load("data/stopParagraphsEmbeddings.npy")
    parrafos = prmTranscripcion.split("\n")
    texto = ""
    for parrafo in parrafos:
        pEmbedding = modelo.encode(parrafo.lower().strip(),convert_to_numpy=True)
        pEmbedding = pEmbedding.reshape(1, -1)
        similitudes = [cosine_similarity(pEmbedding, v.reshape(1, -1)) for v in stopParagraphsEmbeddings]
        if np.max(similitudes)<0.9:
            texto +=parrafo+"\n"
    with open(prmVideoFilePath +".texto.txt", 'w', encoding="utf-8") as newf:
        newf.write(texto )
    return texto


def SpacyNER(prmTexto, prmVideoFilePath):
    model_path = "models/modelo_entrenado_es_core_news_lg"
    nlp = spacy.load(model_path)
    doc = nlp(prmTexto)
    entities = []
    for ent in  doc.ents:
        entity = Segmento(text= ent.text,entity=ent.label_,start=ent.start_char,end=ent.end_char,segments=[])
        entities.append (entity)
    output_data = {"entities":[segmento.to_dict() for segmento in entities]}

    with open(prmVideoFilePath +".SpacyNER.json", 'w', encoding="utf-8") as newf:
        json.dump(output_data, newf, ensure_ascii=False, indent=4)

    return entities


def BetoNER(prmTexto, prmVideoFilePath):
    labels = ["AUTOR", "EVENTO","DESTINO", "MATERIA"]
    model_name = "models/albert-base-8-spanish-finetuned-ner"
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    # Cargar el tokenizador
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Cargar el pipeline de predicción
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    # Texto de prueba
    predictions = ner_pipeline(prmTexto)
    entities = []
    for entity in predictions:
        entity = Segmento(text= entity['word'],entity=entity['entity_group'],start=entity['start'],end=entity['end'],segments=[])
        entities.append (entity)
    output_data = {"entities":[segmento.to_dict() for segmento in entities]}
    with open(prmVideoFilePath +".SpacyBETO.json", 'w', encoding="utf-8") as newf:
        json.dump(output_data, newf, ensure_ascii=False, indent=4)

    return entities


def PrintEnt(prmEnts,level=0):
    if prmEnts==None:
        return
    for ent in prmEnts:
        ident = "-" * level*3
        print(ident,"entity:",ent.entity,"; text:",ent.text, "; Diputado:",ent.Diputado,"; DiputadoId:",ent.DiputadoId,"; Video start",ent.VideoStart)
        PrintEnt(ent.segments,level+1)


def getAutores(prmEntities):
    entities =  copy.deepcopy(prmEntities)
    discursos = []
    segmento : Segmento
    if entities[0].entity != "AUTOR":
       segmento = Segmento(entity="AUTOR",text="",segments=[],start=entities[0].start,end=entities[0].end)
       discursos.append(segmento)

    for ent in entities:
        if  ent.entity!="AUTOR":
            segmento.segments.append(ent)
        else:
            segmento = ent
            segmento.segments=[]
            discursos.append(segmento)
    return discursos

def getOficios(prmEntities):
    if prmEntities==None or len(prmEntities)==0:
        return
    entities =  copy.deepcopy(prmEntities)
    oficios = []
    segmento : Segmento

    if entities[0].entity != "EVENTO":
       segmento = Segmento(entity="EVENTO",text="",segments=[],start=entities[0].start, end=entities[0].end)
       oficios.append(segmento)

    for ent in entities:
        if  ent.entity!="EVENTO":
            segmento.segments.append(ent)
        else:
            segmento = ent
            segmento.segments=[]
            oficios.append(segmento)

    return oficios


def getAutoresData(prmEntities,prmSegments, prmSpeachers):
    fullText = ""
    locations=[]
    for segment in prmSegments["segments"]:
        texto = str(segment["text"]).replace(" ","")
        fullText += texto
        location = (segment["start"],segment["end"])
        locations.extend([location] * len(texto))
    dfDiputados = pd.read_csv('data/Diputados.csv')
    diputadoId=0
    for ent in prmEntities:
        if ent.entity =='AUTOR':
            diputadoId =0
            textoAutor=str(ent.text).replace(" ","")
            if textoAutor=="":
                textoAutor =ent.segments[0].text.replace(" ","")
            indice = fullText.find(textoAutor)
            if indice != -1 and ent.segments!=None and len(ent.segments)>0:
                ent.VideoStart = int(locations[indice][0])
                textoSpeach = ent.segments[0].text.replace(" ","")
                indiceSpeach = fullText.find(textoSpeach,indice)
                if indiceSpeach != -1  :
                    startSpeach = int(locations[indiceSpeach][0])
                    for speach in prmSpeachers:
                        if speach["start"]>=startSpeach: 
                            diputadoId  = speach["closestPerson"]   
                            ent.vi = speach["start"]                  
                            break
                    diputado = dfDiputados[dfDiputados["Id"] == int(diputadoId)]
                    if diputado.empty:
                        diputadoNombre ="NN"
                    else:
                        diputadoNombre = str(diputado["Diputado"].iloc[0])
                    ent.Diputado = diputadoNombre
                    ent.DiputadoId = diputadoId
    return prmEntities


def getDiscursos(prmAutores):
    discursos = []
    for autor in prmAutores:
        oficios = []
        if autor.segments!=None:
            for oficio in autor.segments:
                materia = ""
                destinos = []
                for ent in oficio.segments:
                    if ent.entity =="DESTINO":
                        destinos.append({ "nombre":ent.text})
                    if ent.entity =="MATERIA":
                        materia += ent.text +";"   
                oficioJSON = {"videoInicio":oficio.VideoStart,
                            "destinos" :destinos,
                            "materia" :materia}
                oficios.append(oficioJSON)
            autorJSon = {"diputado" : autor.Diputado,
                        "fotoURL" :"https://www.camara.cl/img.aspx?prmID=GRCL" + str(autor.DiputadoId),
                        "videoInicio": autor.VideoStart,
                        "oficios" : oficios
                        }
            discursos.append(autorJSon)
    return discursos


def SaveOuputFile(prmTitle, prmVideoFilePath, prmModel="spacy"):
    print("Transcripcion Start")
    transcripcionSegments,transcripcionText  = Transcripcion(prmVideoFilePath)
    print("speachers Start")
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