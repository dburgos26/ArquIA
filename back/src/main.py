from typing import Optional
from fastapi import FastAPI, Request, Form, File, UploadFile
from src.graph import graph
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
config = {"configurable": {"thread_id": "1"}}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Origen permitido
    allow_credentials=True,
    allow_methods=["*"],  # Métodos permitidos
    allow_headers=["*"],  # Headers permitidos
)

os.makedirs("images", exist_ok=True)

@app.post("/message")
async def message(message: str = Form(...), file: Optional[UploadFile] = File(None)):
    if not message:
        return {"error": "No message provided"}

    image_path = ""
    if file and file.filename:  # Check if the file is not empty
        # Save the uploaded image
        image_path = os.path.join("images", file.filename)
        with open(image_path, "wb") as image_file:
            content = await file.read()
            image_file.write(content)

    # Prepare the input for the graph
    user_input = message + " " + image_path if image_path else message
    response = graph.invoke({"messages": [user_input]}, config)

    print("--------------------")
    for m in response["messages"]:
        m.pretty_print()
    print("--------------------")

    return {"last_message": response["messages"][-1].content, "messages": response["messages"]}


@app.post("/test")
async def test_endpoint(message: str = Form(...), file: UploadFile = File(None)):
    if not message:
        return {"error": "No message provided"}

    image_path = ""

    user_input = message + " " + image_path if image_path else message
    messages = [user_input]

    return {"last_message": "this is a response to " + user_input, "messages": [{"name": "Supervisor", "text": "Mensaje del supervisor"}, {"name": "reasercher", "text": "Mensaje del investigador"}]}

