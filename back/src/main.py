from typing import Optional
from fastapi import FastAPI, Form, File, UploadFile
from src.graph import graph
import os
import sqlite3
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Origen permitido
    allow_credentials=True,
    allow_methods=["*"],  # Métodos permitidos
    allow_headers=["*"],  # Headers permitidos
)

os.makedirs("images", exist_ok=True)

def get_next_message_id(session_id: str) -> int:
    conn = sqlite3.connect("feedback_db/feedback.db")
    cursor = conn.cursor()
    
    # Get the highest message_id for this session
    cursor.execute(
        "SELECT MAX(message_id) FROM message_feedback WHERE session_id = ?", 
        (session_id,)
    )
    result = cursor.fetchone()[0]
    conn.close()
    
    # If no messages in this session yet, start with 1
    if result is None:
        return 1
    else:
        return result + 1

@app.post("/message")
async def message(
    message: str = Form(...), 
    session_id: str = Form(...),
    image1: Optional[UploadFile] = File(None), 
    image2: Optional[UploadFile] = File(None)
):
    if not message:
        return {"error": "No message provided"}
    
    if not session_id:
        return {"error": "No session ID provided"}

    # Get the next message ID for this session
    message_id = get_next_message_id(session_id)
    
    # Process images
    image_path1 = ""
    if image1 and image1.filename:
        image_path1 = os.path.join("images", image1.filename)
        with open(image_path1, "wb") as image_file:
            content = await image1.read()
            image_file.write(content)

    image_path2 = ""
    if image2 and image2.filename:
        image_path2 = os.path.join("images", image2.filename)
        with open(image_path2, "wb") as image_file:
            content = await image2.read()
            image_file.write(content)

    # Prepare the message list
    messageList = [{"role": "user", "content": message}]

    if image_path1:
        messageList.append({"role": "user", "content": "this is the image path: " + image_path1})

    if image_path2:
        messageList.append({"role": "user", "content": "this is the second image path: " + image_path2})

    config = {"configurable": {"thread_id": str(session_id)}}

    # Prepare the input for the graph
    response = graph.invoke({
        "messages": messageList,
        "userQuestion": message,
        "localQuestion": "",
        "hasVisitedInvestigator": False,
        "hasVisitedDiagrams": False,
        "hasVisitedCreator": False,
        "hasVisitedEvaluator": False,
        "nextNode": "supervisor",
        "imagePath1": image_path1,
        "imagePath2": image_path2,
        "endMessage": "",
        "mermaidCode": ""
    }, config)
    
    # Insert a new record in the feedback table
    conn = sqlite3.connect("feedback_db/feedback.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO message_feedback (session_id, message_id, thumbs_up, thumbs_down) VALUES (?, ?, 0, 0)",
        (session_id, message_id)
    )
    conn.commit()
    conn.close()
    
    # Add session_id and message_id to the response
    response["message_id"] = message_id
    response["session_id"] = session_id
    
    return response

# Rename the endpoint to match what the frontend expects
@app.post("/feedback")
async def feedback(
    session_id: str = Form(...),
    message_id: int = Form(...),
    thumbs_up: int = Form(...),
    thumbs_down: int = Form(...)
):
    conn = sqlite3.connect("feedback_db/feedback.db")
    cursor = conn.cursor()
    
    # Update the feedback values
    cursor.execute(
        "UPDATE message_feedback SET thumbs_up = ?, thumbs_down = ? WHERE session_id = ? AND message_id = ?",
        (thumbs_up, thumbs_down, session_id, message_id)
    )
    conn.commit()
    conn.close()
    
    return {"status": "Feedback recorded successfully"}


@app.post("/test")
async def test_endpoint(message: str = Form(...), file: UploadFile = File(None)):
    if not message:
        return {"error": "No message provided"}

    image_path = ""

    user_input = message + " " + image_path if image_path else message
    messages = [user_input]

    return {"mermaidCode": "classDiagram\n    class Bank {\n        +String name\n        +String address\n        +List~Branch~ branches\n        +createAccount(Customer, double) Account\n        +closeAccount(Account) void\n    }\n\n    class Branch {\n        +String name\n        +String location\n        +List~Customer~ customers\n        +List~Account~ accounts\n        +addCustomer(Customer) void\n        +removeCustomer(Customer) void\n    }\n\n    class Customer {\n        +String name\n        +String address\n        +String phoneNumber\n        +String email\n        +List~Account~ accounts\n        +openAccount(double) Account\n        +closeAccount(Account) void\n    }\n\n    class Account {\n        +String accountNumber\n        +double balance\n        +AccountType accountType\n        +deposit(double) void\n        +withdraw(double) void\n        +getBalance() double\n    }\n\n    class AccountType {\n        <<enumeration>>\n        CHECKING\n        SAVINGS\n        BUSINESS\n    }\n\n    Bank \"1\" --> \"0..*\" Branch\n    Branch \"1\" --> \"0..*\" Customer\n    Customer \"1\" --> \"0..*\" Account\n    Account \"1\" --> \"1\" AccountType\n", "endMessage": "this is a response to " + user_input, "messages": [{"name": "Supervisor", "text": "Mensaje del supervisor"}, {"name": "reasercher", "text": "Mensaje del investigador"}]}

