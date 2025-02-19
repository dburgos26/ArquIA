#%%
from google.cloud import aiplatform
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
import base64
import vertexai
from vertexai.preview.generative_models import Image
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
from back.tools.diagramCreator import run_agent
import uuid

project_id = "tesisdbmarquia"
location = "us-central1"
endpoint_id = "454548002527248384"
endpoint_id2 = "5809978870354411520"

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max = 300)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)



aiplatform.init(project=project_id, location=location)
endpoint = aiplatform.Endpoint(endpoint_id)
endpoint2 = aiplatform.Endpoint(endpoint_id2)

@tool
def diagramCreator(prompt: str) ->str:
    """This tool allows for creation of software architecture diagrams in a XML format.
    Always use this tool when the user wants to create a diagram and be really specific of what you need.
    Here is an example of a valid query: Give me xml code of the diagram of a simple app. I want to have a message broker, connected to two interfaces. I dont have specific attributes or details, just do that"""
    xml_code = run_agent(prompt)
    return xml_code


@tool
def researcher(prompt: str) -> str:
    """This researcher is able of answering questions only about Attribute Driven Design, also known as ADD or ADD 3.0
    Remember the context is software architecture, dont confuse Attribute Driven Design with Attention-Deficit Disorder"""

    project_id = "arquisoftia"
    location_id = "us-central1"
    agent_id = "10651228-ebc3-4845-9147-8ab4617d22d1"
    agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"
    session_id = uuid.uuid4()
    llm = ChatVertexAI(model ="gemini-1.5-flash-002" )
    response = llm.invoke(prompt)
    return response

@tool
def diagram_describer_xml(xml: str) -> str:
    """This tool is used to explain diagrams that are exclusively given by the user in xml code
    Remember to explain thoroughly the purpose of each component of the diagram"""

    generative_multimodal_model = GenerativeModel("gemini-1.5-pro-002")
    response = generative_multimodal_model.generate_content(["Can you describe this xml file?", xml])
    return response
    
@tool
def diagram_describer(image_path: str) -> str:
    """
    This tool is used to explain software architecture diagrams.
    It performs two main tasks:
      - It identifies and explains tactics related to performance and availability (e.g., load balancing and replication).
      - If the diagram is a class diagram, it describes each class, including its attributes, methods, and relationships (such as inheritance, association, or composition).
    The explanation is solely based on the explicit information present in the diagram.
    """
    image = Image.load_from_file(image_path)
    
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    
    prompt = (
        "Analyze the following software architecture diagram. "
        "First, identify and explain any tactics related to performance and availability, such as load balancing and replication. "
        "Additionally, if the diagram is a class diagram (IMPORTANT: ignore the tactics related to performance and availability), describe each class by detailing its attributes, methods, and relationships "
        "(e.g., inheritance, association, composition) and evaluate if the diagram is complete and solves the problem."
        "Focus solely on what is explicitly presented in the diagram."
    )

    print(image_path)
    
    response = generative_multimodal_model.generate_content([prompt, image])
    return response


@tool
def diagram_extraction(image_path: str) -> str:
    """Extracts components from a diagram image. Use when the user says he wants to extract the components of a diagram"""
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()
    encoded_image = base64.b64encode(image_content).decode('utf-8')
    instance = {"content": encoded_image}
    
    try:
        response = endpoint2.predict(instances=[instance])
        print(response)
        return response
    except Exception as e:
        return f"Error during prediction: {str(e)}"
  
@tool   
def classify(image_path: str) -> str:
    """Classifies a Software Architecture Diagram from an image file path."""
    # Load image data
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()
    encoded_image = base64.b64encode(image_content).decode('utf-8')
    # Prepare the request payload for prediction
    instance = {"content": encoded_image}
    
    try:
        # Call the Vertex AI endpoint to get predictions
        response = endpoint.predict(instances=[instance])
        predictions_list = response.predictions[0]
        display_names = predictions_list['displayNames']
        confidences = predictions_list['confidences']

        # Encontrar el índice del valor máximo en confidences
        max_index = confidences.index(max(confidences))

        # Obtener el displayName correspondiente
        best_display_name = display_names[max_index]
        best_confidence = confidences[max_index]
        # Assuming your model returns a category, you can return the most likely one
        if best_display_name == 'Components':
            res = "It is a Software Architecture Components Diagram"
        elif best_display_name == 'Despliegue':
            res = "It is a Software Architecture Deployment Diagram"
        elif best_display_name == 'Contexto':
            res = "It is a Software Architecture Context Diagram"

        return res
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"


tools = [diagramCreator, researcher,diagram_describer_xml,diagram_describer,diagram_extraction,classify, wiki_tool]

import tkinter as tk
from langchain_google_vertexai import ChatVertexAI
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, BaseMessage

memory = MemorySaver()

# Inicializamos el LLM
llm = ChatVertexAI(model="gemini-1.5-flash")

llm_with_tools = llm.bind_tools(tools= tools)
# Definimos la estructura de estado
class State(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(MessagesState)

# Definimos la función del chatbot
def chatBot(state: MessagesState):
    return {"messages": llm_with_tools.invoke(state['messages'])}

def should_continue(state: MessagesState):
    """Return the next node to execute."""
    last_message = state["messages"][-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    # Otherwise if there is, we continue
    return "action"

# Añadimos nodos y bordes al grafo
graph_builder.add_node("chat_bot", chatBot)
graph_builder.add_edge(START, "chat_bot")
tool_node = ToolNode(tools= tools)
graph_builder.add_node("tools", tool_node)


graph_builder.add_conditional_edges(
    "chat_bot",
    tools_condition,
)
graph_builder.add_edge("tools", "chat_bot")
graph_builder.add_edge("chat_bot", END)

# Compilamos el grafo
graph = graph_builder.compile(checkpointer=memory)


#%%
"""
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinter import ttk
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Inicializar el MemorySaver para el grafo
memory = MemorySaver()

# Crear la ventana principal
window = tk.Tk()
window.title("ChatBot")
window.geometry("500x600")
window.configure(bg="#2c3e50")

# Estilos personalizados
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("TLabel", foreground="white", background="#2c3e50", font=("Helvetica", 14))
style.configure("TFrame", background="#2c3e50")

# Frame para contener el chat
chat_frame = ttk.Frame(window)
chat_frame.pack(pady=10, padx=10, fill="both", expand=True)

# Campo de entrada de texto (chat)
input_label = ttk.Label(chat_frame, text="Write a Message:")
input_label.pack(pady=10)


# Área de salida de texto (resultados del chatbot)
output_text = scrolledtext.ScrolledText(chat_frame, height=15, font=("Helvetica", 12), bg="#ecf0f1", fg="#2c3e50", wrap="word")
output_text.pack(pady=10, fill="both", expand=True)

input_text = tk.Text(chat_frame, height=3, font=("Helvetica", 12), wrap="word", bg="#ecf0f1", fg="#2c3e50")
input_text.pack(pady=5, fill="x")

image_status_label = ttk.Label(chat_frame, text="No image has been selected.", font=("Helvetica", 10))
image_status_label.pack(pady=5)


selected_image_path = None


def select_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")],
    )
    if selected_image_path:
        image_status_label.config(text=f"Selected Image: {selected_image_path.split('/')[-1]}")
    else:
        image_status_label.config(text="No image selected")
        messagebox.showwarning("Warning", "No file has been selected.")

def send_message():
    global selected_image_path
    user_message = input_text.get("1.0", "end-1c")
    if not user_message.strip() and not selected_image_path:
        messagebox.showwarning("Warning", "Please, type a message or select an image.")
        return
    
    if selected_image_path:
        user_message += f"\n[Selected Image: {selected_image_path}]"
    selected_image_path = None
    image_status_label.config(text="No image has been selected")
  
    input_text.delete("1.0", "end")
    
    input_message = HumanMessage(content=user_message)
 
    try:
        config = {"configurable": {"thread_id": "2"}}
  
        for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
            bot_response = event["messages"][-1].content
        output_text.insert("end", f"You: {user_message}\nChatbot: {bot_response}\n\n")
        output_text.yview(tk.END) 
        

    except Exception as e:
        messagebox.showerror("Error", str(e))

select_image_button = ttk.Button(chat_frame, text="Select Image", command=select_image)
select_image_button.pack(pady=10)

send_button = ttk.Button(chat_frame, text="Send", command=send_message)
send_button.pack(pady=10)

window.mainloop()
"""