from typing import Optional
from fastapi import FastAPI, Form, File, UploadFile
from src.graph import graph
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
config = {"configurable": {"thread_id": "2"}}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Origen permitido
    allow_credentials=True,
    allow_methods=["*"],  # Métodos permitidos
    allow_headers=["*"],  # Headers permitidos
)

os.makedirs("images", exist_ok=True)

@app.post("/message")
async def message(message: str = Form(...), image1: Optional[UploadFile] = File(None), image2: Optional[UploadFile] = File(None)):
    if not message:
        return {"error": "No message provided"}

    image_path1 = ""
    if image1 and image1.filename:  # Check if the file is not empty
        # Save the uploaded image
        image_path1 = os.path.join("images", image1.filename)
        with open(image_path1, "wb") as image_file:
            content = await image1.read()
            image_file.write(content)

    image_path2 = ""
    if image2 and image2.filename:  # Check if the file is not empty
        # Save the uploaded image
        image_path2 = os.path.join("images", image2.filename)
        with open(image_path2, "wb") as image_file:
            content = await image2.read()
            image_file.write(content)

    messageList = [{"role": "user", "content": message}]

    if image_path1:
        messageList.append({"role": "user", "content": "this is the image path: " + image_path1})

    if image_path2:
        messageList.append({"role": "user", "content": "this is the second image path: " + image_path2})

    

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

    return response


@app.post("/test")
async def test_endpoint(message: str = Form(...), file: UploadFile = File(None)):
    if not message:
        return {"error": "No message provided"}

    image_path = ""

    user_input = message + " " + image_path if image_path else message
    messages = [user_input]

    return {"mermaidCode": "classDiagram\n    class Bank {\n        +String name\n        +String address\n        +List~Branch~ branches\n        +createAccount(Customer, double) Account\n        +closeAccount(Account) void\n    }\n\n    class Branch {\n        +String name\n        +String location\n        +List~Customer~ customers\n        +List~Account~ accounts\n        +addCustomer(Customer) void\n        +removeCustomer(Customer) void\n    }\n\n    class Customer {\n        +String name\n        +String address\n        +String phoneNumber\n        +String email\n        +List~Account~ accounts\n        +openAccount(double) Account\n        +closeAccount(Account) void\n    }\n\n    class Account {\n        +String accountNumber\n        +double balance\n        +AccountType accountType\n        +deposit(double) void\n        +withdraw(double) void\n        +getBalance() double\n    }\n\n    class AccountType {\n        <<enumeration>>\n        CHECKING\n        SAVINGS\n        BUSINESS\n    }\n\n    Bank \"1\" --> \"0..*\" Branch\n    Branch \"1\" --> \"0..*\" Customer\n    Customer \"1\" --> \"0..*\" Account\n    Account \"1\" --> \"1\" AccountType\n", "endMessage": "this is a response to " + user_input, "messages": [{"name": "Supervisor", "text": "Mensaje del supervisor"}, {"name": "reasercher", "text": "Mensaje del investigador"}]}

