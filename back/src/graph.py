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
    hasVisitedCreator: bool
    hasVisitedEvaluator: bool
    nextNode: Literal["investigator", "creator", "evaluator", "unifier"]
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
    nextNode: Literal["investigator", "creator", "evaluator", "unifier"]

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
            "enum": ["investigator", "creator", "evaluator", "unifier"]
        }
    },
    "required": ["setup", "nextNode"]
}

# ========== Prompts 

# ===== Nodes

def makeSupervisorPrompt(state: GraphState) -> str:
    visited_nodes = []
    if state["hasVisitedInvestigator"]:
        visited_nodes.append("investigator")
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

    These are the possible outputs: ['investigator', 'creator', 'evaluator', 'unifier'].
    In case there is nothing else to do go to unifier

    You also need to define a specific question for the node you send: 
        - for the investigator node only ask for concepts or patterns in the user diagrams
        - for the creator node ask to generate a diagram or code example
        - for the evaluator node ask to evaluate the user's ideas, IF THE USER HAVE 2 DIAGRAMS YOU NEED TO SEND TO THIS NODE ONLY
    always give some context in the question for the specific node
    """ 

    return supervisorPrompt

prompt_researcher = """You are an expert in software architecture, specializing in Attribute Driven Design (ADD) and tactics related to availability 
    and performance. Your task is to analyze the user's question and provide an accurate and well-explained response based on your expertise. 
    You have access to two tools to assist you in answering:

    - 'LLM': A powerful large language model fine-tuned for software architecture-related queries. Use this tool when you need 
      a detailed explanation, best practices, or general knowledge about architecture principles.
    - 'LLMWithImages': A large language model with image support, allowing you to analyze diagrams, patterns, and visual representations

"""
# TODO add rag tool
"""
    
    - 'local_RAG': A local Retrieval-Augmented Generation (RAG) system with access to a curated knowledge base of software architecture 
      documentation, case studies, and academic papers. Use this tool when you need precise, contextually relevant, or document-backed 
      answers.
"""

prompt_creator ="""You are an expert in creating xml code for software architecture diagrams.
You are working with a researcher who will tell you exactly what you should do, please follow their instructions. Limit 
yourself to load balancing and replication. You yourself are unable to create this diagrams, but you have a tool for
this purpose named diagramCreator. Always use this tool  when the user asks for creation, modifications, or additions to the diagram"""

evaluatorPrompt = f"""You are an expert in software architecture evaluation, specializing in assessing project feasibility and analyzing 
    the strengths and weaknesses of proposed strategies. Your role is to critically evaluate the user's request and provide a well-informed 
    assessment based on two specialized tools:

    - `Theory Tool` for correctness checks.
    - `Viability Tool` for feasibility assessment.
    - `Needs Tool` for requirement alignment.
    - `Analyze Tool` for comparing two diagrams.
    """

# ===== Tools

llm_prompt = "Retrieve general software architecture knowledge. Answer concisely and focus on key concepts:"

llmWithImages_prompt = """Analyze the diagram and provide a detailed explanation of the software architecture tactics found in the image. 
    Focus on performance and availability tactics"""

# TODO add rag tool

# TODO add diagram creator tool

theory_prompt = "Analyze the theoretical correctness of this architecture diagram. Follow best practices."

viability_prompt = "Evaluate the feasibility of the user's ideas. Provide a detailed analysis of the viability of the proposed strategies."

needs_prompt = "Analyze the user's requirements and check if they align with the proposed architecture. Focus on the user's needs."

analyze_prompt = """Analyze the following pair of diagrams:
    A class diagram representing the implementation of a component,
    A component diagram that places this component within the architectural context.
    Evaluate whether the component's implementation (class diagram) is properly designed to support quality attributes such as scalability and 
    performance, among others.
    Provide a detailed assessment highlighting strengths, deficiencies, and improvement suggestions."""

# ========== Tools 

# ===== Investigator

@tool
def LLM(prompt: str) -> str:
    """This researcher is able of answering questions only about Attribute Driven Design, also known as ADD or ADD 3.0
    Remember the context is software architecture, dont confuse Attribute Driven Design with Attention-Deficit Disorder"""

    response = llm.invoke(prompt)
    return response

@tool
def LLMWithImages(image_path: str) -> str:
    """This researcher is able of answering questions about software architecture diagrams, patterns, and visual representations.
    Remember to focus on performance and availability tactics, and always use the image as a reference"""

    image = Image.load_from_file(image_path)
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    response = generative_multimodal_model.generate_content([
        "What software architecture tactics can you see in this diagram? "
        "If it is a class diagram, analyze and evaluate it by identifying classes, attributes, methods, relationships, "
        "Object-Oriented Design principles, and design patterns.", 
        image
    ])
    return response

# TODO add rag tool    

# ===== Creator

@tool
def diagram_creator(prompt: str) ->str:
    """This tool allows for creation of software architecture diagrams in a XML format.
    Always use this tool when the user wants to create a diagram and be really specific of what you need.
    Here is an example of a valid query: Give me xml code of the diagram of a simple app. I want to have 
    a message broker, connected to two interfaces. I dont have specific attributes or details, just do that"""
    xml_code = run_agent(prompt)
    return xml_code

# ===== Evaluator

@tool
def theory_tool(prompt: str) -> str:
    """This evaluator is able to check the theoretical correctness of the architecture diagram. It follows best practices and provides a detailed analysis."""

    response = llm.invoke(theory_prompt + prompt)
    return response

@tool
def viability_tool(prompt: str) -> str:
    """This evaluator is able to check the feasibility of the user's ideas. It provides a detailed analysis of the viability of the proposed strategies."""

    response = llm.invoke(viability_prompt + prompt)
    return response

@tool
def needs_tool(prompt: str) -> str:
    """This evaluator is able to check the user's requirements and verify if they align with the proposed architecture. It focuses on the user's needs."""

    response = llm.invoke(needs_prompt + prompt)
    return response

@tool
def analyze_tool(image_path: str, image_path2: str) -> str:
    """This evaluator is able to compare two diagrams: a class diagram representing the implementation 
    of a component and a component diagram that places this component within the architectural context. 
    It evaluates whether the component's implementation (class diagram) is properly designed to support 
    quality attributes such as scalability and performance, among others. It provides a detailed assessment 
    highlighting strengths, deficiencies, and improvement suggestions."""

    image = Image.load_from_file(image_path)
    image2 = Image.load_from_file(image_path2)
    generative_multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
    response = generative_multimodal_model.generate_content([
        analyze_prompt,
        image,
        image2
    ])
    return response

# ========== Router 

def router(state: GraphState) -> Literal["investigator", "creator", "evaluator", "unifier"]:
    if state["nextNode"] == "unifier":
        return "unifier"
    elif state["nextNode"] == "investigator" and not state["hasVisitedInvestigator"]:
        return "investigator"
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
    
researcher_agent = create_react_agent(llm, tools=[LLM, LLMWithImages],state_modifier=prompt_researcher)

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

evaluator_agent = create_react_agent(llm, tools=[theory_tool, viability_tool, needs_tool], state_modifier=evaluatorPrompt)

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
builder.add_node("creator", creator_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("unifier", unifier_node)

# ========== Edges creation 

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", router)
builder.add_edge("investigator", "supervisor")
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