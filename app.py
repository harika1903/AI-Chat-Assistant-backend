# app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import httpx  # Using httpx for async requests
import os
import json

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Global Constants ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"
STATIC_DIR = "static"

# --- Mounting Static Files ---
# This serves all files from your "static" directory (e.g., index.html, css, js)
# It assumes you have an index.html file in a folder named "static".
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- API Endpoints ---

@app.get("/")
def serve_homepage():
    """
    Serves the main index.html file from the 'static' directory.
    """
    # Make sure you have an 'index.html' file in your 'static' folder.
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/chat")
async def chat(prompt: str = Query(..., description="User prompt for AI model")):
    """
    Handles the chat logic. It receives a prompt, sends it to the Ollama model,
    and streams the response back to the client.
    """
    data_payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True  # Enable streaming from Ollama
    }

    async def stream_generator():
        """
        This async generator function streams the response from Ollama.
        It now correctly handles JSON objects that may be split across multiple chunks.
        """
        buffer = ""
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", OLLAMA_URL, json=data_payload) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        # Process buffer line by line
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        yield data["response"]
                                except json.JSONDecodeError:
                                    print(f"Skipping invalid JSON line: {line}")
                                    continue
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            yield f"Error connecting to Ollama: {e}"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            yield f"An unexpected error occurred: {e}"

    # Return a StreamingResponse, which FastAPI handles efficiently.
    return StreamingResponse(stream_generator(), media_type="text/plain")


# --- Main Execution Block ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server on http://localhost:8000")
    # To run this file, save it as app.py and use the command: uvicorn app:app --reload
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
