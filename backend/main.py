"""
FastAPI application to serve the HackerNews Score Prediction model.
"""
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

# Import the prediction function from our source code
from src.predict import predict_score

app = FastAPI(
    title="HackerNews Score Predictor API",
    description="An API to predict the potential score of a HackerNews submission.",
    version="1.0.0"
)

# Define the request body model using Pydantic for data validation
class Story(BaseModel):
    title: str
    url: str  # Using str as HttpUrl can be too strict for some HN links
    user: str
    timestamp: int = int(datetime.now().timestamp())

    class Config:
        schema_extra = {
            "example": {
                "title": "Show HN: I built a tool to predict HN scores",
                "url": "https://github.com/myuser/myproject",
                "user": "myuser",
                "timestamp": 1678886400
            }
        }

# Define the response body model
class PredictionResponse(BaseModel):
    predicted_score: int

@app.get("/", tags=["Root"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the HackerNews Score Predictor API!"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def post_predict(story: Story):
    """
    Predicts the score of a HackerNews story based on its features.
    
    - **title**: The title of the story.
    - **url**: The URL of the story. Can be empty for self-posts.
    - **user**: The username of the submitter.
    - **timestamp**: The UNIX timestamp of the submission.
    """
    score = predict_score(
        title=story.title,
        url=story.url,
        user=story.user,
        timestamp=story.timestamp
    )
    return {"predicted_score": score}

# To run this app:
# uvicorn backend.main:app --reload 