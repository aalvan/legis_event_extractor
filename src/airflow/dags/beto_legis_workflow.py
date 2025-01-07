from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.models import Variable
#from src.legis_workflow.utils import toSpacy, SpacytoConLL
from datetime import datetime, timedelta
import json

default_args = {
    'owner': 'Alexis Alva',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retry': False,
}

SHARED_DATA = 'opt/airflow/data/processed/conll_dataset.json'

@task.virtualenv(
    task_id="prepare_data",
    requirements=["spacy==3.8.3"],
    venv_cache_path="/tmp/venv_cache"
)
def _preprocess_data():
    import spacy
    import json
    from spacy.cli import download

    FILE_PATH = "/opt/airflow/data/dataset.json"
    dataset = []
    with open(FILE_PATH, "r") as file:
        lines = file.readlines()
        for line in lines:
            data = json.loads(line)
            dataset.append(data)

    def to_spacy(dataSet):
        spacy_data = []
        for entry in dataSet:
            text = str(entry['text'])
            entities = [(start, end, label) for start, end, label in entry['label']]
            spacy_data.append((text, {"entities": entities}))
        return spacy_data


    def spacy_to_conll(dataset):
        """
        Convierte un archivo JSON con texto y etiquetas a formato CoNLL usando spaCy.
        Args:
            input_file (str): Ruta del archivo JSON de entrada.
            model (str): Modelo de spaCy para tokenización.
        """
        # Cargar el modelo de spaCy
        try:
            nlp = spacy.load("es_core_news_lg")
        except OSError:
            print("es_core_news_lg not found. Downloading the model...")
            download("es_core_news_lg")

        conLLData = []
        # Abrir el archivo de salida
        for entry in dataset:
            text = entry[0]
            labels = entry[1]["entities"]
            connTexto=""
            # Procesar el texto con spaCy
            doc = nlp(text)

                # Inicializar etiquetas BIO
            tags = ["O"] * len(doc)

            # Asignar etiquetas según las entidades
            for start, end, label in labels:
                for token in doc:
                    if token.idx >= start and token.idx < end:
                        if token.idx == start:
                            tags[token.i] = f"B-{label}"  # Inicio de la entidad
                        else:
                            tags[token.i] = f"I-{label}"  # Dentro de la entidad

                # Escribir cada token y su etiqueta en el archivo de salida
            for token, tag in zip(doc, tags):
                if token.text.strip()=='':
                    connTexto += "\n"
                else:
                    connTexto += f"{token.text} {tag}\n"

            conLLData.append(connTexto)
        return conLLData
    
    def align_offsets_to_tokens(doc, entities):
        """
        Alinea los offsets de las entidades a los límites de los tokens generados por spaCy.
        """
        aligned_entities = []
        for start, end, label in entities:
            token_start = None
            token_end = None
            for token in doc:
                # Encontrar el token que contiene el inicio de la entidad
                if token.idx <= start < token.idx + len(token.text):
                    token_start = token.idx
                # Encontrar el token que contiene el final de la entidad
                if token.idx < end <= token.idx + len(token.text):
                    token_end = token.idx + len(token.text)
            # Si ambos límites están definidos, añadir la entidad ajustada
            if token_start is not None and token_end is not None:
                aligned_entities.append((token_start, token_end, label))
        return aligned_entities

    spacy_dataset = to_spacy(dataset)

    # para arreglar las etiquetas que inician/terminan en espacios en blanco
    i = 0
    for text, props in spacy_dataset:
        for index, row in enumerate(props["entities"]):
            start = row[0]
            end = row[1]
            while text[start] == " ":
                start += 1
                i += 1

            while text[end - 1] == " ":
                end -= 1
                i += 1

            my_list = list(row)

            # Modificar un elemento
            my_list[0] = start
            my_list[1] = end

            # Volver a convertir a tupla si es necesario
            props["entities"][index] = tuple(my_list)
    print(i)

    nlp = spacy.blank("es")

    for text, props in spacy_dataset:
        doc = nlp.make_doc(text)
        props["entities"] = align_offsets_to_tokens(doc, props["entities"])

    conll_dataset = spacy_to_conll(spacy_dataset)
    with open(save_path, "w") as file:
        json.dump(conll_dataset, file)

    print(f'Saved processed dataset to {SHARE_DATA}')

@task
def _tokenize_and_prepare_dataset(**kwargs):
    with open(f"{SHARE_DATA}", "r") as file:
        conll_dataset = json.load

    dataset, labels, id2label, label2id = load_custom_dataset(conLLDataset)

    print(id2label)
    print(label2id)
    print(labels)


with DAG('beto_legis_workflow',
    default_args=default_args,
    description='BERTO Legis Workflow',
    start_date=None,
    schedule_interval=None,
    tags=['mlops']
    ) as dag:

    preprocess_data = _preprocess_data()
    tokenize_and_prepare_dataset = _tokenize_and_prepare_dataset()

    preprocess_data >> tokenize_and_prepare_dataset