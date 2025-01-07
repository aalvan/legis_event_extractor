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
    nlp = spacy.load("es_core_news_lg")

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