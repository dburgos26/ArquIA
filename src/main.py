from fastapi import FastAPI, Request
from src.graph import graph
import os

app = FastAPI()
config = {"configurable": {"thread_id": "1"}}


@app.post("/message")
async def message(request: Request):
    body = await request.json()
    msg = body.get("message")

    if not msg:
        return {"error": "No message provided"}
    
    response = graph.invoke({"messages": msg},config)

    print("--------------------")
    for m in response["messages"]:
        m.pretty_print()
    print("--------------------")

    return response

