from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn


from named_entity_recognition import NERecognition

ner = NERecognition()

torch.set_num_threads(1)

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/recognize")
def recognize_entities(request: TextRequest):
    entities = ner.recognize(request.text)
    return {"entities": entities}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6363, workers=2)