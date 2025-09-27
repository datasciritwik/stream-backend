import os
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()

# --- CORS Middleware ---
# This is crucial for allowing your HTML file (on a different "origin")
# to communicate with your backend server.
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "null",  # Important for opening the HTML file directly
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Define the directory where recordings will be saved
SAVE_DIRECTORY = Path("recordings")

@app.on_event("startup")
async def startup_event():
    """Create the save directory on server startup if it doesn't exist."""
    SAVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    print(f"Recordings will be saved to: {SAVE_DIRECTORY.resolve()}")

@app.post("/upload-audio/")
async def upload_audio(audio: UploadFile = File(...)):
    """
    Receives an audio file and saves it to the specified directory
    with a unique, timestamped filename.
    """
    try:
        # Generate a unique filename using a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"utterance_{timestamp}.wav"
        save_path = SAVE_DIRECTORY / filename

        # Save the uploaded file
        # shutil.copyfileobj is efficient for streaming file content
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)

        return {
            "status": "success",
            "filename": filename,
            "saved_path": str(save_path)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        # Close the file to release resources
        audio.file.close()
        
@app.get("/status")
async def status():
    return "On"