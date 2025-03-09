# ========== Imports 

# Util
from typing_extensions import TypedDict
from typing import Annotated, Literal
import os
from dotenv import load_dotenv, find_dotenv
from src.diagramCreator import run_agent

# langchain
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END

# GCP
from vertexai.generative_models import GenerativeModel
from vertexai.preview.generative_models import Image
from google.cloud import aiplatform

# ========== Start

load_dotenv(dotenv_path=find_dotenv('.env.development'))

project_id = os.getenv('PROJECT_ID')
location = os.getenv('LOCATION')
endpoint_id = os.getenv('ENDPOINT_ID')
endpoint_id2 = os.getenv('ENDPOINT_ID2')

memory = MemorySaver()

llm = ChatOpenAI(model="gpt-4o")

class GraphState(TypedDict):
    messages: list
    userQuestion: str
    localQuestion: str
    hasVisitedInvestigator: bool
    hasVisitedDiagrams: bool
    hasVisitedCreator: bool
    hasVisitedEvaluator: bool
    nextNode: Literal["investigator", "diagrams", "creator", "evaluator", "unifier"]
    imagePath: str
    endMessage: str

class AgentState(TypedDict):
    messages: list
    userQuestion: str
    localQuestion: str
    imagePath: str
    
builder = StateGraph(GraphState)

class supervisorResponse(TypedDict):
    localQuestion: Annotated[str, ..., "What is the cuestion for the worker node?"]
    nextNode: Literal["investigator", "diagrams", "creator", "evaluator", "unifier"]

supervisorSchema = {
    "title": "SupervisorResponse",
    "description": "Response from the supervisor indicating the next node and the setup question.",
    "type": "object",
    "properties": {
        "localQuestion": {
            "type": "string",
            "description": "What is the question for the worker node?"
        },
        "nextNode": {
            "type": "string",
            "description": "The next node to act.",
            "enum": ["investigator", "diagrams", "creator", "evaluator", "unifier"]
        }
    },
    "required": ["setup", "nextNode"]
}

# ========== Prompts 

def makeSupervisorPrompt(state: GraphState) -> str:
    visited_nodes = []
    if state["hasVisitedInvestigator"]:
        visited_nodes.append("investigator")
    if state["hasVisitedDiagrams"]:
        visited_nodes.append("diagrams")
    if state["hasVisitedCreator"]:
        visited_nodes.append("creator")
    if state["hasVisitedEvaluator"]:
        visited_nodes.append("evaluator")

    visited_nodes_str = ", ".join(visited_nodes) if visited_nodes else "none"

    supervisorPrompt = f"""You are a supervisor tasked with managing a conversation between the following workers: investigator, 
    diagrams, creator, evaluator. Given the following user request, respond with the worker to act next. Each worker will perform 
    a task and respond with their results and status. There are 4 possible nodes: the investigator that has access to LLM and a local 
    RAG, the diagrams node that classifies and extracts, the creator node that generates an image or code, and the evaluator 
    that checks for viability and correctness of what the user says. There are some important flows you must respect: if the user just wants to 
    know about Attribute Driven Design, only use the investigator. If the user wants to extract elements of a diagram, you must 
    always classify first that diagram. If the user wants to describe a diagram, you must always first extract the elements of 
    said diagram. If the user wants to create a diagram, you must always first research about the software architecture tactics 
    that will be implemented in the created diagram. These are the nodes that you have visited: {visited_nodes_str}.
    
    This is the user question: {state["userQuestion"]}

    These are the possible outputs: ['investigator', 'diagrams', 'creator', 'evaluator', 'unifier'].
    In cas there is nothing else to do go to unifier
    """ 

    return supervisorPrompt

prompt_researcher = """You are an expert in software architecture, specializing in Attribute Driven Design (ADD) and tactics related to availability 
    and performance. Your task is to analyze the user's question and provide an accurate and well-explained response based on your expertise. 
    You have access to two tools to assist you in answering:

    - 'specialized_LLM': A powerful large language model fine-tuned for software architecture-related queries. Use this tool when you need 
      a detailed explanation, best practices, or general knowledge about architecture principles.

"""
# TODO add rag tool
"""
    
    - 'local_RAG': A local Retrieval-Augmented Generation (RAG) system with access to a curated knowledge base of software architecture 
      documentation, case studies, and academic papers. Use this tool when you need precise, contextually relevant, or document-backed 
      answers.
"""

prompt_diagrams = """You are an expert in software architecture diagrams, specializing in classification, description, and extraction of relevant 
    components. Your role is to analyze the given diagram and determine the best approach to process it based on the user's request. 

    You have access to the following tools:

    - 'diagram_classifier': Identifies the type of diagram (e.g., UML, C4, sequence diagram, component diagram) to determine how it should be processed.
    - 'diagram_descriptor': Generates a textual description of the diagram, summarizing its main elements and relationships.
    - 'diagram_extractor': Extracts detailed elements (e.g., components, connections, attributes) from the diagram for further analysis or transformation.
    """

prompt_creator ="""You are an expert in creating xml code for software architecture diagrams.
You are working with a researcher who will tell you exactly what you should do, please follow their instructions. Limit 
yourself to load balancing and replication. You yourself are unable to create this diagrams, but you have a tool for
this purpose named diagramCreator. Always use this tool  when the user asks for creation, modifications, or additions to the diagram"""

evaluatorPrompt = f"""You are an expert in software architecture evaluation, specializing in assessing project feasibility and analyzing 
    the strengths and weaknesses of proposed strategies. Your role is to critically evaluate the user's request and provide a well-informed 
    assessment based on two specialized tools:

    - 'feasibility_checker': Analyzes whether the proposed project, system, or architecture is viable based on technical, resource, and 
      operational constraints.
    - 'strategy_analyzer': Evaluates the proposed architectural strategy, listing its pros and cons to help the user make informed decisions.
    """

# ========== Tools 

# ===== Investigator

@tool
def researcher_LLM(prompt: str) -> str:
    """This researcher is able of answering questions only about Attribute Driven Design, also known as ADD or ADD 3.0
    Remember the context is software architecture, dont confuse Attribute Driven Design with Attention-Deficit Disorder"""

    response = llm.invoke(prompt)
    return response

# TODO add rag tool

# ===== Diagrams

@tool
def diagram_describer(image_path: str) -> str:
    """This tool is used exclusively to explain the software architecture tactics found in an imageof a diagram given by the user. 
    Don't assume anything that is not explictly in the diagram, and focus most of all in tactics associated with performance and 
    availability (specifically Load Balancing and Replication tactics).If there are no explicit tactics associated to this, 
    you must say so"""

    image = Image.load_from_file(image_path)
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    response = generative_multimodal_model.generate_content([
        "What software architecture tactics can you see in this diagram? "
        "If it is a class diagram, analyze and evaluate it by identifying classes, attributes, methods, relationships, "
        "Object-Oriented Design principles, and design patterns.", 
        image
    ])
    return response

@tool
def diagram_classify(image_path: str) -> str:
    """Classifies a Software Architecture Diagram from an image file path using a LLM."""
    
    try:
        image = Image.load_from_file(image_path)
        model = GenerativeModel("gemini-1.0-pro-vision")
        
        response = model.generate_content([
            "What type of software architecture diagram is this? Choose one of: "
            "Components, Deployment, Context or Class. Only return the classification not any other text.",
            image
        ])
        
        classification = response.text.strip()

        if "Components" in classification:
            return "It is a Software Architecture Components Diagram"
        elif "Deployment" in classification:
            return "It is a Software Architecture Deployment Diagram"
        elif "Class" in classification:
            return "It is a Software Architecture Class Diagram"
        elif "Context" in classification:
            return "It is a Software Architecture Context Diagram"
        else:
            return "I couldn't determine the type of diagram."

    except Exception as e:
        return f"Error during classification: {str(e)}"
    
@tool
def diagram_extractor(image_path: str, diagram_type: str) -> str:
    """Extracts structured information from a software architecture diagram.
    
    - If the diagram is a class diagram, it extracts:
      - List of identified classes.
      - Attributes of each class.
      - Methods of each class.
      - Relationships between classes (association, inheritance, aggregation, composition, dependency).
      
    - If the diagram is a component or deployment diagram, it provides:
      - A high-level description of components or infrastructure.
      - Focus on performance and availability tactics (load balancing, replication, etc.).
    
    Use this tool when analyzing UML class diagrams, component diagrams, or deployment diagrams.
    """
    
    try:
        image = Image.load_from_file(image_path)
        model = GenerativeModel("gemini-1.0-pro-vision")
        
        if diagram_type.lower() == "class":
            prompt = (
                "Analyze this UML class diagram and extract:\n"
                "- List of identified classes.\n"
                "- Attributes of each class.\n"
                "- Methods of each class.\n"
                "- Relationships between classes (association, inheritance, aggregation, composition, dependency)."
            )
        else:
            prompt = (
                "Analyze this software architecture diagram and describe:\n"
                "- The key components and their roles.\n"
                "- If it is a deployment diagram, describe the infrastructure elements.\n"
                "- Any performance or availability tactics applied (e.g., load balancing, replication)."
            )
        
        response = model.generate_content([prompt, image])
        return response.text

    except Exception as e:
        return f"Error during extraction: {str(e)}"
    
    

# ===== Creator

@tool
def diagram_creator(prompt: str) ->str:
    """This tool allows for creation of software architecture diagrams in a XML format.
    Always use this tool when the user wants to create a diagram and be really specific of what you need.
    Here is an example of a valid query: Give me xml code of the diagram of a simple app. I want to have a message broker, connected to two interfaces. I dont have specific attributes or details, just do that"""
    xml_code = run_agent(prompt)
    return xml_code

# ===== Evaluator

@tool
def feasibility_checker(prompt: str) -> str:
    """This tool is able of evaluate the viability of the plans the usser suggests, it can tell if the users ideas are viable or if it needs
    any adjusments"""

    response = llm.invoke(prompt)
    return response

@tool
def strategy_analyzer(prompt: str) -> str:
    """This tools is able judge the pros and cons of the user's ideas, it is able to give a detailed analysis of the user's ideas
    and give a detailed analysis of the pros and cons of the user's ideas"""

    response = llm.invoke(prompt)
    return response

# ========== Router 

def router(state: GraphState) -> Literal["investigator", "diagrams", "creator", "evaluator", "unifier"]:
    if state["nextNode"] == "unifier":
        return "unifier"
    elif state["nextNode"] == "investigator" and not state["hasVisitedInvestigator"]:
        return "investigator"
    elif state["nextNode"] == "diagrams" and not state["hasVisitedDiagrams"]:
        return "diagrams"
    elif state["nextNode"] == "creator" and not state["hasVisitedCreator"]:
        return "creator"
    elif state["nextNode"] == "evaluator" and not state["hasVisitedEvaluator"]:
        return "evaluator"
    else:
        return "unifier"

# ========== Nodes definition 

# ===== Supervisor

def supervisor_node(state: GraphState):
    message = [
            {"role": "system", "content": makeSupervisorPrompt(state)},
        ] 
    
    response = llm.with_structured_output(supervisorSchema).invoke(message)

    state_updated: GraphState = {
        **state,
        "localQuestion": response["localQuestion"],
        "nextNode": response["nextNode"]
    }

    return state_updated

# ===== Investigator
    
researcher_agent = create_react_agent(llm, tools=[researcher_LLM],state_modifier=prompt_researcher)

def researcher_node(state: GraphState) -> GraphState:
    result = researcher_agent.invoke(
        {
            "messages": state["messages"],
            "userQuestion": state["userQuestion"],
            "localQuestion": state["localQuestion"],
            "imagePath": state["imagePath"]
        }
    )

    return {
        **state,
        "messages": [AIMessage(content=msg.content, name="researcher") for msg in result["messages"]],
        "hasVisitedInvestigator": True
    }

# ===== Diagrams

diagrams_agent = create_react_agent(llm, tools=[diagram_classify, diagram_extractor, diagram_describer], state_modifier=prompt_diagrams)

def diagrams_node(state: GraphState) -> GraphState:
    result = diagrams_agent.invoke(
        {
            "messages": state["messages"],
            "userQuestion": state["userQuestion"],
            "localQuestion": state["localQuestion"],
            "imagePath": state["imagePath"]
        }
    )

    return {
        **state,
        "messages": [AIMessage(content=msg.content, name="diagrams") for msg in result["messages"]],
        "hasVisitedDiagrams": True
    }

# ===== Creator

creator_agent = create_react_agent(llm, tools=[diagram_creator], state_modifier=prompt_creator)

def creator_node(state: GraphState) -> GraphState:
    result = creator_agent.invoke(
        {
            "messages": state["messages"],
            "userQuestion": state["userQuestion"],
            "localQuestion": state["localQuestion"],
            "imagePath": state["imagePath"]
        }
    )

    return {
        **state,
        "messages": [AIMessage(content=msg.content, name="creator") for msg in result["messages"]],
        "hasVisitedCreator": True
    }

# ===== Evaluator

evaluator_agent = create_react_agent(llm, tools=[feasibility_checker, strategy_analyzer], state_modifier=evaluatorPrompt)

def evaluator_node(state: GraphState) -> GraphState:
    result = evaluator_agent.invoke(
        {
            "messages": state["messages"],
            "userQuestion": state["userQuestion"],
            "localQuestion": state["localQuestion"],
            "imagePath": state["imagePath"]
        }
    )

    return {
        **state,
        "messages": [AIMessage(content=msg.content, name="evaluator") for msg in result["messages"]],
        "hasVisitedEvaluator": True
    }

# ===== Unifier

def unifier_node(state: GraphState) -> GraphState:
    prompt = f"""You are an expert assistant in information synthesis. You will receive a list of messages that may contain 
    scattered ideas, arguments, questions, and answers. Your task is to unify and structure the information into several 
    coherent paragraphs, ensuring clarity and fluency for the user. Each paragraph should be limited to one idea. This is 
    the list of messages you need to unify: {state['messages']}"""

    response = llm.invoke(prompt)

    return {
        **state,
        "endMessage": response.content
    }

# ========== Nodes creation 

builder.add_node("supervisor", supervisor_node)
builder.add_node("investigator", researcher_node)
builder.add_node("diagrams", diagrams_node)
builder.add_node("creator", creator_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("unifier", unifier_node)

# ========== Edges creation 

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", router)
builder.add_edge("investigator", "supervisor")
builder.add_edge("diagrams", "supervisor")
builder.add_edge("creator", "supervisor")
builder.add_edge("evaluator", "supervisor")
builder.add_edge("unifier", END)

# ========== Graph 

graph = builder.compile(checkpointer=memory)

"""
config = {"configurable": {"thread_id": "1"}}

from PIL import Image

graph_image_path = "graph.png"
graph_image = graph.get_graph().draw_mermaid_png()
with open(graph_image_path, "wb") as f:
    f.write(graph_image)

test = graph.invoke({
        "messages": [] ,
        "userQuestion": "What is add 3.0", 
        "localQuestion": "", 
        "hasVisitedInvestigator": False
        , "hasVisitedDiagrams": False, 
        "hasVisitedCreator": False, 
        "hasVisitedEvaluator": False, 
        "nextNode": "supervisor", 
        "imagePath": "",
        "endMessage": ""
    }, config)


print(test)
"""