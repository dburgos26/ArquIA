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
    nextNode: Literal["investigator", "creator", "evaluator", "asr", "unifier"]
    imagePath1: str
    imagePath2: str
    endMessage: str
    hasVisitedASR: bool 

class AgentState(TypedDict):
    messages: list
    userQuestion: str
    localQuestion: str
    imagePath1: str
    imagePath2: str
    
builder = StateGraph(GraphState)

class supervisorResponse(TypedDict):
    localQuestion: Annotated[str, ..., "What is the question for the worker node?"]
    nextNode: Literal["investigator", "creator", "evaluator", "asr", "unifier"]

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
            "enum": ["investigator", "creator", "evaluator", "unifier", "asr"]
        }
    },
    "required": ["localQuestion", "nextNode"]
}

class evaluatorResponse(TypedDict):
    positiveAspects: Annotated[str, ..., "What are the positive aspects of the user's idea?"]
    negativeAspects: Annotated[str, ..., "What are the negative aspects of the user's idea?"]
    suggestions: Annotated[str, ..., "What are the suggestions for improvement?"]

evaluatorSchema = {
    "title": "EvaluatorResponse",
    "description": "Response from the evaluator indicating the positive and negative aspects of the user's idea.",
    "type": "object",
    "properties": {
        "positiveAspects": {
            "type": "string",
            "description": "What are the positive aspects of the user's idea?"
        },
        "negativeAspects": {
            "type": "string",
            "description": "What are the negative aspects of the user's idea?"
        },
        "suggestions": {
            "type": "string",
            "description": "What are the suggestions for improvement?"
        }
    },
    "required": ["positiveAspects", "negativeAspects", "suggestions"]
}

class investigatorResponse(TypedDict):
    definition: Annotated[str, ..., "What is the definition of the concept?"]
    useCases: Annotated[str, ..., "What are the use cases of the concept?"]
    examples: Annotated[str, ..., "What are the examples of the concept?"]

investigatorSchema = {
    "title": "InvestigatorResponse",
    "description": "Response from the investigator indicating the definition, use cases, and examples of the concept.",
    "type": "object",
    "properties": {
        "definition": {
            "type": "string",
            "description": "What is the definition of the concept?"
        },
        "useCases": {
            "type": "string",
            "description": "What are the use cases of the concept?"
        },
        "examples": {
            "type": "string",
            "description": "What are the examples of the concept?"
        }
    },
    "required": ["definition", "useCases", "examples"]
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
    if state.get("hasVisitedASR", False):
        visited_nodes.append("asr")

    visited_nodes_str = ", ".join(visited_nodes) if visited_nodes else "none"

    supervisorPrompt = f"""You are a supervisor tasked with managing a conversation between the following workers: investigator, 
    diagrams, creator, evaluator, and ASR advisor. Given the following user request, respond with the worker to act next. 
    Each worker will perform a task and respond with their results and status.
    
    Important flows:
    - If the user just asks about ADD, use the investigator.
    - If the user wants to extract or describe a diagram, first classify or extract the diagram elements.
    - If the user provides an ASR and limitations (and optionally an image of an implementation), route to the ASR node.
    
    Visited nodes so far: {visited_nodes_str}.
    
    This is the user question: {state["userQuestion"]}
    
    The possible outputs: ['investigator', 'creator', 'evaluator', 'asr', 'unifier'].

    In case there is nothing else to do go to unifier.
    
    You also need to define a specific question for the node you send:
      - For the investigator node: ask for concepts or patterns in the user diagrams.
      - For the creator node: ask to generate a diagram or code example.
      - **For the ASR node: always prioritize it if the user provides an ASR and limitations in the question.**
        - If no diagram is provided, ask for recommendations on how to implement the ASR given the limitations.  
        - If a diagram is provided, ask to evaluate whether the implementation meets the ASR and adheres to the limitations.
      - For the evaluator node: ask to evaluate the user's ideas, especially if two diagrams are provided.
    
    NOTE: YOU CANNOT GO DIRECTLY TO THE UNIFIER NODE; you must go to at least one worker node before.
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
this purpose named diagramCreator. Always use this tool when the user asks for creation, modifications, or additions to the diagram.
"""

def getEvaluatorPrompt(image_path1: str, image_path2) -> str:
    image1 = ""
    image2 = ""

    if image_path1:
        image1 = "this is the first image path: " + image_path1
    if image_path2:
        image2 = "this is the second image path: " + image_path2

    evaluatorPrompt = f"""You are an expert in software architecture evaluation, specializing in assessing project feasibility and analyzing 
        the strengths and weaknesses of proposed strategies. Your role is to critically evaluate the user's request and provide a well-informed 
        assessment based on two specialized tools:

        - `Theory Tool` for correctness checks.
        - `Viability Tool` for feasibility assessment.
        - `Needs Tool` for requirement alignment.
        - `Analyze Tool` for comparing two diagrams.
        {image1}
        {image2}
        """
    return evaluatorPrompt

# ===== Tools

llm_prompt = "Retrieve general software architecture knowledge. Answer concisely and focus on key concepts:"

llmWithImages_prompt = """Analyze the diagram and provide a detailed explanation of the software architecture tactics found in the image. 
    Focus on performance and availability tactics."""

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
    """This researcher is able of answering questions only about Attribute Driven Design, also known as ADD or ADD 3.0.
    Remember the context is software architecture, don't confuse Attribute Driven Design with Attention-Deficit Disorder."""
    response = llm.with_structured_output(investigatorSchema).invoke(prompt)
    return response

@tool
def LLMWithImages(image_path: str) -> str:
    """This researcher is able of answering questions about software architecture diagrams, patterns, and visual representations.
    Remember to focus on performance and availability tactics, and always use the image as a reference."""
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
def diagram_creator(prompt: str) -> str:
    """This tool allows for creation of software architecture diagrams in an XML format.
    Always use this tool when the user wants to create a diagram and be really specific about what you need.
    Here is an example of a valid query: Give me XML code of the diagram of a simple app. I want to have 
    a message broker connected to two interfaces. I don't have specific attributes or details; just do that."""
    xml_code = run_agent(prompt)
    return xml_code

# ===== Evaluator

@tool
def theory_tool(prompt: str) -> str:
    """This evaluator is able to check the theoretical correctness of the architecture diagram. It follows best practices and provides a detailed analysis."""
    response = llm.with_structured_output(evaluatorSchema).invoke(theory_prompt + prompt)
    return response

@tool
def viability_tool(prompt: str) -> str:
    """This evaluator is able to check the feasibility of the user's ideas. It provides a detailed analysis of the viability of the proposed strategies."""
    response = llm.with_structured_output(evaluatorSchema).invoke(viability_prompt + prompt)
    return response

@tool
def needs_tool(prompt: str) -> str:
    """This evaluator is able to check the user's requirements and verify if they align with the proposed architecture. It focuses on the user's needs."""
    response = llm.with_structured_output(evaluatorSchema).invoke(needs_prompt + prompt)
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

def router(state: GraphState) -> Literal["investigator", "creator", "evaluator", "asr", "unifier"]:
    if state["nextNode"] == "unifier":
        return "unifier"
    elif state["nextNode"] == "asr" and not state.get("hasVisitedASR", False):
        return "asr"
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
    
researcher_agent = create_react_agent(llm, tools=[LLM, LLMWithImages], state_modifier=prompt_researcher)

def researcher_node(state: GraphState) -> GraphState:
    result = researcher_agent.invoke({
        "messages": state["messages"],
        "userQuestion": state["userQuestion"],
        "localQuestion": state["localQuestion"],
        "imagePath1": state["imagePath1"],
        "imagePath2": state["imagePath2"]
    })
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg.content, name="researcher") for msg in result["messages"]],
        "hasVisitedInvestigator": True
    }

# ===== Creator

creator_agent = create_react_agent(llm, tools=[diagram_creator], state_modifier=prompt_creator)

def creator_node(state: GraphState) -> GraphState:
    result = creator_agent.invoke({
        "messages": state["messages"],
        "userQuestion": state["userQuestion"],
        "localQuestion": state["localQuestion"],
        "imagePath1": state["imagePath1"],
        "imagePath2": state["imagePath2"]
    })
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg.content, name="creator") for msg in result["messages"]],
        "hasVisitedCreator": True
    }

# ===== Evaluator

def evaluator_node(state: GraphState) -> GraphState:
    evaluator_agent = create_react_agent(llm, tools=[theory_tool, viability_tool, needs_tool, analyze_tool], 
                                          state_modifier=getEvaluatorPrompt(state["imagePath1"], state["imagePath2"]))
    result = evaluator_agent.invoke({
        "messages": state["messages"],
        "userQuestion": state["userQuestion"],
        "localQuestion": state["localQuestion"],
        "imagePath1": state["imagePath1"],
        "imagePath2": state["imagePath2"]
    })
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg.content, name="evaluator") for msg in result["messages"]],
        "hasVisitedEvaluator": True
    }

# ===== Unifier

def unifier_node(state: GraphState) -> GraphState:
    prompt = f"""You are an expert assistant in information synthesis. You will receive a list of messages that may contain 
    scattered ideas, arguments, questions, and answers.

    YOUR MOST IMPORTANT TASK: Format your response using MULTIPLE CLEARLY SEPARATED PARAGRAPHS.
    
    Specific requirements:
    1. Organize the information into 2-4 distinct paragraphs minimum.
    2. Each paragraph must focus on exactly one main idea or theme.
    3. Use proper paragraph breaks (double line breaks) between paragraphs.
    4. Never merge all content into a single paragraph.
    5. Prioritize readability through clear structure over completeness.
    
    Synthesize this information into a well-structured response: {state['messages']}"""
    response = llm.invoke(prompt)
    return {
        **state,
        "endMessage": response.content
    }

# ===== ASR

def asr_node(state: GraphState) -> GraphState:
    if state["imagePath1"]:
        prompt = f"""You are an expert in software architecture implementation evaluation.
            The user has provided the following details:
            {state["userQuestion"]}

            An implementation diagram is available at: {state["imagePath1"]}.

            Evaluate whether the implementation meets the requirements and respects the limitations mentioned in the user question.
            Provide detailed feedback and suggestions for improvement if needed.
            Also it is **IMPORTANT** to analize the implementation of architectural tactics in the user diagram. If there are no architectural tactics in the diagram, please recommend which architectural tactics can be applied.
            """
        result = llm.invoke(prompt)
        message = AIMessage(content=result.content, name="asr_evaluator")
    else:
        prompt = f"""You are an expert in providing recommendations for software architecture.
            The user has provided the following details:
            {state["userQuestion"]}.

            Provide clear recommendations and a step-by-step guide on how to implement the requirement considering the limitations mentioned in the user question.
            It is **IMPORTANT** that in the answer you mention architectural tactics that are important in the implementation of ASR.
            """
        result = llm.invoke(prompt)
        message = AIMessage(content=result.content, name="asr_recommender")
    
    return {
        **state,
        "messages": state["messages"] + [message],
        "hasVisitedASR": True
    }

# ========== Nodes creation 

builder.add_node("supervisor", supervisor_node)
builder.add_node("investigator", researcher_node)
builder.add_node("creator", creator_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("unifier", unifier_node)
builder.add_node("asr", asr_node)

# ========== Edges creation 

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", router)
builder.add_edge("investigator", "supervisor")
builder.add_edge("creator", "supervisor")
builder.add_edge("evaluator", "supervisor")
builder.add_edge("unifier", END)
builder.add_edge("asr", "supervisor")

# ========== Graph 

graph = builder.compile(checkpointer=memory)

"""
config = {"configurable": {"thread_id": "1"}}

from PIL import Image

graph_image_path = "graph.png"
graph_image = graph.get_graph().draw_mermaid_png()
with open(graph_image_path, "wb") as f:
    f.write(graph_image)

# Updated test invocation with correct keys:
test = graph.invoke({
    "messages": [],
    "userQuestion": "What is ADD 3.0? Provide recommendations for its implementation under budget constraints.",
    "localQuestion": "",
    "hasVisitedInvestigator": False,
    "hasVisitedCreator": False,
    "hasVisitedEvaluator": False,
    "hasVisitedASR": False,
    "nextNode": "supervisor",
    "imagePath1": "",  # No image provided
    "imagePath2": "",
    "endMessage": ""
}, config)

print(test)
"""
