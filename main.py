from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")
def predict(title: str):
    # Load the model
    model = load_model()

    # Preprocess the title
    title_embedding = preprocess_title(title)

    # Make a prediction
    prediction = model.predict(title_embedding) 

    # Return the prediction
    return {"prediction": prediction}

def load_model():
    return None

def preprocess_title(title: str):
    return None