from typing_extensions import TypedDict
from typing import Literal
from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import Image
from langchain_core.tools import tool
from src.diagramCreator import run_agent
from pydantic import BaseModel
import base64
from google.cloud import aiplatform
from dotenv import load_dotenv, find_dotenv
import os
#from IPython.display import Image, display

load_dotenv(dotenv_path=find_dotenv('.env.development'))

project_id = os.getenv('PROJECT_ID')
location = os.getenv('LOCATION')
endpoint_id = os.getenv('ENDPOINT_ID')
endpoint_id2 = os.getenv('ENDPOINT_ID2')

aiplatform.init(project=project_id, location=location)
endpoint = aiplatform.Endpoint(endpoint_id)
endpoint2 = aiplatform.Endpoint(endpoint_id2)

members = ["creator",  "researcher", "extractor", "classifier"]
options = members + ["FINISH"]

class AgentState(MessagesState):
    """The 'next' variable indicates where to route to next."""
    next: str 


#AGENT PROMPTS

def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINISH so the team knows to stop."
        f"\n{suffix}"
    )

prompt_researcher = """You are a software architecture expert who knows about Attribute Driven Design (also
known as ADD or ADD 3.0) and about tactics related to availability and performance (specifically load balancing and replication of servers/databases).
If you are asked a puntual question about ADD or these tactics. If you are asked something like: Do you know about tactics for software architecture?, 
ANSWER YOURSELF AND DO NOT CONTACT THE CREATOR AGENT
You must also be able to answer questions about load balancing and replication, once you answer include FINISH. 
ALWAYS FINISH THE JOB AFTER ANSWERING THE QUESTION. Always be really specific of what you need when directing the diagram creator and always indicate that you dont want any other specifications and details.
Always ask a single simple text query similar to this example: Give me xml code of the diagram of a simple app. I want to have a load balancer, connected to two Application servers and a client. I dont have specific attributes or details, just do that.
If the user asks for modifications of the diagram, you can just replicate the query, such as: Add a third server.
IF the user previously asked for a modification, please include it in the new query.
YOU MUST ALWAYS DELEGATE THE MODIFICATIONS OF THE XML CODE TO THE CREATOR, NEVER MODIFY YOURSELF THE XML CODE
OF THE DIAGRAMS
"""

prompt_classifier = """You are an expert at classifying software architecure diagram. You are working with an expert extracting components of a diagram.
Make sure to let him know the classification you made. You will receive the path to the image, give that to the tool
in order to classify the diagrams. Pass the path of the image to the extractor too. NEVER include FINISH unless the user has
only asked you to classify the diagram."""


prompt_extractor ="""You can extract the components out of diagrams, limit yourself to that function.
You are working with a diagram classifier and a component describer. You will recieve the classification of the diagram. Make sure to tell the component describer
what components you found, and remember to tell them the describer tool which classification the classifier did.
You will receive the path to the image, give that to the describer too. Make sure to make the describer tool go next, it does not
matter if your result is too vague"""

prompt_describer ="""You are an expert describing the components of a software architecture diagram, focusing in
tactics related to availability and performance of the system. You are working with a diagram extractor that will provide
you with the classification of the diagram and the elements that he extracted. Make sure to take into consideration this 
elements in your answer, and describe extensively the elements provided that relate to scalability and
performance tactics (in particular, load balancing and replication)"""

prompt_creator ="""You are an expert in creating xml code for software architecture diagrams.
You are working with a researcher who will tell you exactly what you should do, please follow their instructions. Limit 
yourself to load balancing and replication. You yourself are unable to create this diagrams, but you have a tool for
this purpose named diagramCreator. Always use this tool  when the user asks for creation, modifications, or additions to the diagram"""

@tool
def diagram_describer(image_path: str) -> str:
    """This tool is used exclusively to explain the software architecture tactics found in an image
    of a diagram given by the user. Don't assume anything that is not explictly in the diagram, and 
    focus most of all in tactics associated with performance and availability (specifically Load Balancing and Replication tactics).
    If there are no explicit tactics associated to this, you must say so"""
    image = Image.load_from_file(image_path)

    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    response = generative_multimodal_model.generate_content(["What software architecture tactics can you see in this diagram?", image])
    return response

@tool
def diagramCreator(prompt: str) ->str:
    """This tool allows for creation of software architecture diagrams in a XML format.
    Always use this tool when the user wants to create a diagram and be really specific of what you need.
    Here is an example of a valid query: Give me xml code of the diagram of a simple app. I want to have a message broker, connected to two interfaces. I dont have specific attributes or details, just do that"""
    xml_code = run_agent(prompt)
    return xml_code

@tool
def diagram_describer_xml(xml: str) -> str:
    """This tool is used to explain diagrams that are exclusively given by the user in xml code
    Remember to explain thoroughly the purpose of each component of the diagram"""

    generative_multimodal_model = GenerativeModel("gemini-1.5-pro-002")
    response = generative_multimodal_model.generate_content(["Can you describe this xml file?", xml])
    return response

@tool
def researcher(prompt: str) -> str:
    """This researcher is able of answering questions only about Attribute Driven Design, also known as ADD or ADD 3.0
    Remember the context is software architecture, dont confuse Attribute Driven Design with Attention-Deficit Disorder"""

    llm = ChatVertexAI(model ="gemini-1.5-flash-002" )
    response = llm.invoke(prompt)
    return response

@tool
def diagram_extraction(image_path: str) -> str:
    """Extracts components from a diagram image. 
    Use when the user says he wants to extract the components of a diagram"""
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()
    encoded_image = base64.b64encode(image_content).decode('utf-8')
    # Prepare the request payload for prediction
    instance = {"content": encoded_image}

    print("-----------------2")
    
    try:
        # Call the Vertex AI endpoint to get predictions
        response = endpoint2.predict(instances=[instance])
        return response
    except Exception as e:
        return f"Error during prediction: {str(e)}"


@tool
def classify(image_path: str) -> str:
    """Classifies a Software Architecture Diagram from an image file path."""
    with open(image_path, "rb") as image_file:
        image_content = image_file.read()
    encoded_image = base64.b64encode(image_content).decode('utf-8')
    instance = {"content": encoded_image}
    try:
        response = endpoint.predict(instances=[instance])

        print("-----------------")
        print("classifier")
        print(response)
        print("-----------------")

        predictions_list = response.predictions[0]
        display_names = predictions_list['displayNames']
        confidences = predictions_list['confidences']
        max_index = confidences.index(max(confidences))
        best_display_name = display_names[max_index]

        if best_display_name == 'Components':
            res = "It is a Software Architecture Components Diagram"
        elif best_display_name == 'Despliegue':
            res = "It is a Software Architecture Deployment Diagram"
        elif best_display_name == 'Contexto':
            res = "It is a Software Architecture Context Diagram"

        return res
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"




class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal["creator", "researcher", "extractor", "classifier", "FINISH"]

def router(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if "FINISH" in last_message.content:
        return END
    return "continue"


def make_supervisor_node(llm: BaseChatModel, members: list[str]) -> str:
    system_prompt = (
        """You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. There are some important
        flows you must respect: if the user just wants to know about Attribute Driven Design, only use 
        the researcher. If the user wants to extract elements of a diagram, you must
        always classify first that diagram, if the user wants to describe a diagram,
        you must always first extract the elements of said diagram. If the user wants to create
        a diagram, you must always first research about the software architecture tactics that 
        will be implemented in the created diagram.
        """

    )



    def supervisor_node(state: AgentState) -> AgentState:
        """An LLM-based router."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = llm.with_structured_output(Router).invoke(messages) 
        next_ =   response.next
        if next_ == "FINISH":
            next_ = END

        return {"next": next_}

    return supervisor_node

memory = MemorySaver()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

builder = StateGraph(MessagesState)


#Creacion agentes expertos

describer_agent = create_react_agent(llm, tools=[diagram_describer],state_modifier=make_system_prompt(prompt_describer))

def describer_node(state: AgentState) -> AgentState:
    result = describer_agent.invoke(state)
    return {
        "messages": [
            HumanMessage(content=result["messages"][-1].content, name="describer")
        ]
    }


creator_agent = create_react_agent(llm, tools=[diagramCreator], state_modifier=make_system_prompt(prompt_creator))

def creator_node(state: AgentState) -> AgentState:
    result = creator_agent.invoke(state)
    return {
        "messages":[
            HumanMessage(content=result["messages"][-1].content, name="creator")
        ]
    }

researcher_agent = create_react_agent(llm, tools=[researcher], state_modifier= make_system_prompt(prompt_researcher))

def researcher_node(state: AgentState) -> AgentState:
    result = researcher_agent.invoke(state)
    return {
        "messages":[
            HumanMessage(content=result["messages"][-1].content, name = "researcher")
        ]
    }

extraction_agent = create_react_agent(llm, tools=[diagram_extraction], state_modifier=make_system_prompt(prompt_extractor))

def extraction_node(state: AgentState) -> AgentState:
    result = extraction_agent.invoke(state)
    return {
        "messages":[
            HumanMessage(content=result["messages"][-1].content, name="extractor")
        ]
    }

classifier_agent = create_react_agent(llm, tools=[classify], state_modifier=make_system_prompt(prompt_classifier))

def classify_node(state: AgentState) -> AgentState:
    result = classifier_agent.invoke(state)
    return {
        "messages":[
            HumanMessage(content=result["messages"][-1].content, name="classifier")
        ]
    }


#Node creation

supervisor_node = make_supervisor_node(llm, members)
builder.add_node("supervisor", supervisor_node)
builder.add_node("describer", describer_node)
builder.add_node("creator", creator_node)
builder.add_node("researcher", researcher_node)
builder.add_node("extractor", extraction_node)
builder.add_node("classifier", classify_node)
builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", END)


#Edges Creation



builder.add_conditional_edges("supervisor", lambda state: state["next"])
builder.add_conditional_edges("classifier", router, {"continue": "extractor", END: END})
builder.add_edge("extractor","describer")
builder.add_conditional_edges("researcher", router, {"continue": "creator", END: END})
builder.add_edge("creator", END)
builder.add_edge("describer", END)


graph = builder.compile(checkpointer=memory)


#image_data = graph.get_graph(xray=True).draw_mermaid_png()


#with open("graph_image.png", "wb") as f:
#    f.write(image_data)