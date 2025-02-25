import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from back.chatBotVersion2 import graph
from langchain_core.messages import HumanMessage
from PIL import Image

# Constants
height = 600
title = "MultiAgent Software Architecture Assistant"
icon = ":robot:"
config = {"configurable": {"thread_id": "2"}}

def save_uploaded_file(uploaded_file, save_dir="data"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def generate_message(user_input, class_image_path=None, component_image_path=None):
    # Si se tienen ambas imágenes, las incluimos en el payload
    if class_image_path and component_image_path:
        # El payload puede ser un diccionario que incluya tanto el mensaje como la ruta de ambas imágenes.
        payload = {
            "messages": [user_input, class_image_path, component_image_path]
        }
        # Usamos stream para enviar el payload
        for event in graph.stream(payload, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
            response = event["messages"][-1].content
        ai_messages = response
    else:
        # Si no se cargaron las imágenes, se envía solo el mensaje de texto
        for event in graph.stream({"messages": [user_input]}, config, stream_mode="values"):
            print(event["messages"][-1])
            response = event["messages"][-1].content
        ai_messages = response
        
    # Guardamos la conversación en el estado de sesión
    st.session_state.conversation.append({
        "user": user_input,
        "analyst": ai_messages,
        "class_image_path": class_image_path,
        "component_image_path": component_image_path
    })

    # Mostrar la conversación
    for entry in st.session_state.conversation:
        st.write(f"**You**: {entry['user']}")
        if entry.get("class_image_path"):
            st.image(entry["class_image_path"], caption="Diagrama de Clases", use_column_width=True)
        if entry.get("component_image_path"):
            st.image(entry["component_image_path"], caption="Diagrama de Componentes", use_column_width=True)
        st.write(f"**Analyst**: {entry['analyst']}")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.set_page_config(page_title=title, page_icon=icon)
st.header(title)

# Dos file uploaders: uno para el diagrama de clases y otro para el diagrama de componentes
uploaded_file_class = st.file_uploader("Selecciona el diagrama de clases", type=["jpg", "jpeg", "png"], key="class")
uploaded_file_component = st.file_uploader("Selecciona el diagrama de componentes", type=["jpg", "jpeg", "png"], key="component")

file_path_class = None
file_path_component = None

if uploaded_file_class is not None:
    file_path_class = save_uploaded_file(uploaded_file_class)

if uploaded_file_component is not None:
    file_path_component = save_uploaded_file(uploaded_file_component)

# Input de texto para el prompt
if prompt := st.chat_input("Enter Prompt..."):
    generate_message(prompt, class_image_path=file_path_class, component_image_path=file_path_component)
