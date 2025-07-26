# app.py
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Configuration ---
# This allows your frontend (on GitHub Pages) to communicate with your backend (on Render).
origins = ["*"] # Allows all origins for simplicity

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- Initialize Groq Client ---
# The API key will be read from the environment variable we set on Render.
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    client = None

# --- API Endpoint ---
@app.post("/chat")
async def chat(prompt: str = Query(..., description="User prompt for AI model")):
    """
    Handles the chat logic by sending the prompt to the Groq API
    and streaming the response back.
    """
    async def stream_generator():
        if not client:
            yield "Error: AI service is not configured. The API key may be missing."
            return

        try:
            # Use Groq's streaming API
            stream = await client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-8b-8192", # A fast and capable model on Groq
                stream=True,
            )
            # Yield each piece of the response as it arrives
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as e:
            print(f"An error occurred with the Groq API: {e}")
            yield "Sorry, an error occurred while connecting to the AI service."

    return StreamingResponse(stream_generator(), media_type="text/plain")

# A simple root endpoint to confirm the server is running
@app.get("/")
def read_root():
    return {"status": "Backend is running and ready to accept chat requests."}

# This part is for local testing and not used by Render
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server for local testing on http://localhost:8000")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
