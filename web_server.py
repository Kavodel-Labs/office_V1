
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging

# In-memory message store for simplicity
# In a real application, this would be backed by a database or the memory system.
messages = []

# --- Pydantic Models for API data validation ---
class UserMessage(BaseModel):
    content: str

class SystemMessage(BaseModel):
    sender: str
    content: str

# --- FastAPI Application ---
app = FastAPI()
logger = logging.getLogger("aethelred_web")

# This will be set by the main application to get access to the Secretary agent
secretary_agent = None

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serves the main HTML user interface."""
    try:
        with open("local_ui.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: local_ui.html not found.</h1>", status_code=500)

@app.post("/send_message")
async def send_message(message: UserMessage):
    """Receives a message from the user and passes it to the system."""
    logger.info(f"Received message from UI: {message.content}")
    messages.append({"sender": "User", "content": message.content})

    if secretary_agent:
        # Create a task for the secretary agent to process
        task = {
            'type': 'process_incoming_message',
            'source': 'local_ui',
            'payload': {
                'user': 'local_user',
                'text': message.content
            }
        }
        await secretary_agent.process_task(task)
    else:
        # Fallback if the secretary agent is not available
        await asyncio.sleep(1) # Simulate processing time
        placeholder_response = f"Aethelred has received your message: '{message.content}' (Secretary not available)"
        messages.append({"sender": "Aethelred", "content": placeholder_response})

    return {"status": "message received"}

@app.get("/get_updates")
async def get_updates():
    """Provides the latest messages to the UI."""
    return {"messages": messages}

def set_secretary_agent(agent):
    """Allows the main application to inject the secretary agent."""
    global secretary_agent
    secretary_agent = agent

if __name__ == "__main__":
    import uvicorn
    print("Starting Aethelred Web UI server...")
    print("Open http://127.0.0.1:8000 in your browser.")
    uvicorn.run(app, host="127.0.0.1", port=8000)
