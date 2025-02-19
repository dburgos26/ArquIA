import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from back.chatBot import graph

from langchain_core.messages import HumanMessage
import os
from PIL import Image

# Constants
height = 600
title = "MultiAgent Software Architecture Assistant"
icon = ":robot"
config = {"configurable": {"thread_id": "2"}}
file_path = None

def generate_message(user_input, image_path):
    global file_path

    if image_path:

        for event in graph.stream({"messages": [user_input+image_path]}, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
            response = event["messages"][-1].content
        ai_messages = response
    else:
        for event in graph.stream({"messages": [user_input]}, config, stream_mode="values"):
            print(event["messages"][-1])
            response = event["messages"][-1].content
        ai_messages = response
        
    st.session_state.conversation.append({
        "user": user_input,
        "analyst": ai_messages,
        "image_path": image_path
    })

    for entry in st.session_state.conversation:
        st.write(f"**You**: {entry['user']}")
        if entry.get("image_path"):
            st.image(entry["image_path"], caption="Uploaded Image", use_column_width=True)
        st.write(f"**Analyst**: {entry['analyst']}")

    file_path = None


if "conversation" not in st.session_state:
    st.session_state.conversation = []


st.set_page_config(page_title=title, page_icon=icon)
st.header(title)

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    save_dir = "data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, uploaded_file.name)


    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

if prompt := st.chat_input("Enter Prompt.."):
   
    if file_path:
        generate_message(prompt, image_path=file_path)
    else:
        generate_message(prompt, image_path=None)



