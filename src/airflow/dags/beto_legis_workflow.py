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
    PROCESSED_DATA_PATH = '/opt/airflow/data/processed/conll_dataset.json'

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
             nlp= spacy.load("es_core_news_lg")
        except OSError:
            print("es_core_news_lg not found. Downloading the model...")
            download("es_core_news_lg")
            nlp= spacy.load("es_core_news_lg")

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
    with open(PROCESSED_DATA_PATH, "w") as file:
        json.dump(conll_dataset, file)

    print(f'Saved processed dataset to {PROCESSED_DATA_PATH}')

@task.virtualenv(
    task_id="tokenize_and_prepare_dataset",
    requirements=["datasets", "torch"],
    venv_cache_path="/tmp/venv_cache"
)
def _tokenize_and_prepare_dataset():
    import json
    from datasets import load_dataset, DatasetDict, Dataset

    PROCESSED_DATA_PATH = '/opt/airflow/data/processed/conll_dataset.json'
    SAVE_PATH = '/opt/airflow/data/processed/'

    with open(f"{PROCESSED_DATA_PATH}", "r") as file:
        conll_dataset = json.load(file)

    def load_custom_dataset(data_list):
        """
        Convierte una lista de strings en un DatasetDict compatible con Hugging Face.
        Args:
            data_list (list): Lista de strings en formato CoNLL.
        Returns:
            DatasetDict: Conjunto de datos dividido en entrenamiento y prueba.
        """
        sentences = []
        ner_tags = []
        current_sentence = []
        current_tags = []

        for doc in data_list:
            for line in doc.split("\n"):
                if line.strip() == "":  # Nueva oración
                    if current_sentence:
                        sentences.append(current_sentence)
                        ner_tags.append(current_tags)
                        current_sentence = []
                        current_tags = []
                else:
                    token, tag = line.strip().split()
                    current_sentence.append(token)
                    current_tags.append(tag)

            # Añadir la última oración
            if current_sentence:
                sentences.append(current_sentence)
                ner_tags.append(current_tags)

        # Convertir etiquetas BIO a índices numéricos
        unique_tags = sorted(set(tag for tags in ner_tags for tag in tags))
        tag2id = {tag: i for i, tag in enumerate(unique_tags)}
        id2tag = {i: tag for tag, i in tag2id.items()}

        # Transformar etiquetas en índices
        ner_tags = [[tag2id[tag] for tag in tags] for tags in ner_tags]

        # Crear un DatasetDict
        dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {"tokens": sentences[: int(0.8 * len(sentences))], "ner_tags": ner_tags[: int(0.8 * len(sentences))]}
                ),
                "test": Dataset.from_dict(
                    {"tokens": sentences[int(0.8 * len(sentences)) :], "ner_tags": ner_tags[int(0.8 * len(sentences)) :]}
                ),
            }
        )

        return dataset, unique_tags, id2tag, tag2id

    dataset, labels, id2label, label2id = load_custom_dataset(conll_dataset)

    dataset.save_to_disk(f"{SAVE_PATH}dataset")
    with open(f"{SAVE_PATH}labels.json", "w") as file:
        json.dump(labels, file)
    with open(f"{SAVE_PATH}id2label.json", "w") as file:
        json.dump(id2label, file)
    with open(f"{SAVE_PATH}label2id.json", "w") as file:
        json.dump(label2id, file)

    print(f"Variables saved to {SAVE_PATH}")
    print(id2label)
    print(label2id)
    print(labels)

@task.virtualenv(
    task_id="train_model",
    requirements=["transformers", "datasets", "scikit-learn", "torch"],
    venv_cache_path="/tmp/venv_cache"
)
def _train_model():
    import json
    from sklearn.metrics import classification_report
    from datasets import load_from_disk
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
    )
    SAVE_PATH = '/opt/airflow/data/processed/'
    MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
    OUTPUT_DIR = './data/ner_output'

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    dataset = load_from_disk(f'{SAVE_PATH}dataset')
    with open(f"{SAVE_PATH}labels.json", "r") as file:
        labels = json.load(file)
    with open(f"{SAVE_PATH}id2label.json", "r") as file:
        id2label = json.load(file)
    with open(f"{SAVE_PATH}label2id.json", "r") as file:
        label2id = json.load(file)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(label[word_id])
                else:
                    label_ids.append(-100)
                previous_word_id = word_id
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    """training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
    )
    def compute_metrics(pred):
        labels = pred.label_ids.flatten()
        preds = np.argmax(pred.predictions, axis=2).flatten()
        true_labels = [id2label[label] for label in labels if label != -100]
        true_preds = [id2label[pred] for pred, label in zip(preds, labels) if label != -100]
        report = classification_report(true_labels, true_preds, output_dict=True, zero_division=0)
        return {
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1": report["weighted avg"]["f1-score"],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))"""

with DAG('beto_legis_workflow',
    default_args=default_args,
    description='BERTO Legis Workflow',
    start_date=None,
    schedule_interval=None,
    tags=['mlops']
    ) as dag:

    preprocess_data = _preprocess_data()
    tokenize_and_prepare_dataset = _tokenize_and_prepare_dataset()
    train_model = _train_model()

    preprocess_data >> tokenize_and_prepare_dataset >> train_model