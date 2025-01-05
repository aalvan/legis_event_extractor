def toSpacy(dataSet):
    spacy_data = []
    for entry in dataSet:
        text = str(entry['text']) 
        entities = [(start, end, label) for start, end, label in entry['label']]
        spacy_data.append((text, {"entities": entities}))
    return spacy_data

def SpacytoConLL(dataset):
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
