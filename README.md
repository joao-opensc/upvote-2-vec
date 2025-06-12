# HackerNews Score Prediction

This project contains a machine learning model to predict the potential score of a HackerNews submission. The model is built with PyTorch and served via a FastAPI application. A Streamlit application is also included to interact with the model.

This repository is a refactored version of an experimental Jupyter Notebook, organized for maintainability and deployment.

## Project Structure

```
.
├── artifacts/                # Stores trained model, encoders, scalers
├── backend/                  # FastAPI application
│   └── main.py
├── data/                     # Raw data (not checked into git)
├── notebooks/                # Experimental notebooks
├── src/                      # Main source code
│   ├── config.py
│   ├── data_processing.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── streamlit_app.py          # Streamlit frontend application
├── Dockerfile                # To containerize the application
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Data:**
    Place your `hackernews_full_data.parquet` file inside the `data/` directory. The GloVe embeddings will be downloaded automatically on the first run if not found.

## Training the Model

To train the model from scratch, run the training script. This will process the data, train the model, and save the resulting artifacts (model weights, encoders, scaler) to the `artifacts/` directory.

```bash
python -m src.train
```

This process is tracked using Weights & Biases. Make sure you are logged in (`wandb login`) if you want to sync the results.

## Running the Backend API

Once the model is trained and the artifacts are in the `artifacts/` directory, you can serve the model via the FastAPI application.

### Locally with Uvicorn

For development, you can run the app directly with Uvicorn, which supports live reloading.

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive documentation at `http://127.0.0.1:8000/docs`.

### With Docker

To run the application in a containerized environment, first build the Docker image:

```bash
docker build -t hn-score-predictor .
```

Then, run the container:

```bash
docker run -p 8000:8000 hn-score-predictor
```

The API will be accessible at `http://localhost:8000`.

## Running the Frontend

With the backend API running, you can start the Streamlit frontend.

```bash
streamlit run streamlit_app.py
```

The application will be available at a local URL, typically `http://localhost:8501`.

## Using the API

You can send a `POST` request to the `/predict` endpoint with your story's data.

**Example using `curl`:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "title": "Show HN: I built a tool to predict HN scores",
  "url": "https://github.com/myuser/myproject",
  "user": "myuser",
  "timestamp": 1678886400
}'
```

**Expected Response:**

```json
{
  "predicted_score": 123
}
``` 