class Segmento:
    def __init__(self, entity : str, text: str, start : int, end :int,segments ):
        self.entity = entity
        self.text = text
        self.start = start
        self.end = end
        self.segments = segments
        self.Diputado =""
        self.DiputadoId=-1
        self.VideoStart=0

    def to_dict(self):
            return {
                "entity": self.entity,
                "text": self.text,
                "start": self.start,
                "end": self.end,
                "entities":[segmento.to_dict() for segmento in self.segments]
            }

    @classmethod
    def from_dict(cls, data):
        return cls(data["entity"], data["text"], data["start"], data["end"])

    def __repr__(self):
        return f"Segmento(Entidad='{self.entity}', Texto='{self.text}', Inicio={self.start}, fin={self.end},entities:[{[segmento.to_dict() for segmento in self.segments]}])"
