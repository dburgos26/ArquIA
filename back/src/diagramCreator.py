import uuid
from google.cloud import dialogflowcx as df
import os


def run_agent(text: str):
    # TODO(developer): Replace these values when running the function
    project_id = "arquisoftia"
    # For more information about regionalization see https://cloud.google.com/dialogflow/cx/docs/how/region
    location_id = "us-central1"
    # For more info on agents see https://cloud.google.com/dialogflow/cx/docs/concept/agent
    agent_id = "10651228-ebc3-4845-9147-8ab4617d22d1"
    agent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"
    # For more information on sessions see https://cloud.google.com/dialogflow/cx/docs/concept/session
    session_id = uuid.uuid4()
    # For more supported languages see https://cloud.google.com/dialogflow/es/docs/reference/language
    language_code = "en-us"

    response = detect_intent_texts(agent, session_id, text, language_code)
    parsed_response = extract_xml(response)

    save_to_drawio_file(parsed_response)

    return parsed_response


def detect_intent_texts(agent, session_id, text, language_code):
    """Returns the result of detect intent with texts as inputs.

    Using the same `session_id` between requests allows continuation
    of the conversation."""
    session_path = f"{agent}/sessions/{session_id}"
    print(f"Session path: {session_path}\n")
    client_options = None
    agent_components = df.AgentsClient.parse_agent_path(agent)
    location_id = agent_components["location"]
    if location_id != "global":
        api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
        print(f"API Endpoint: {api_endpoint}\n")
        client_options = {"api_endpoint": api_endpoint}
    session_client = df.SessionsClient(client_options=client_options)

    
    text_input = df.TextInput(text=text)
    query_input = df.QueryInput(text=text_input, language_code=language_code)
    request = df.DetectIntentRequest(
        session=session_path, query_input=query_input
    )
    response = session_client.detect_intent(request=request)

    response_messages = [
        " ".join(msg.text.text) for msg in response.query_result.response_messages
    ]
    return response_messages[0]


def extract_xml(response_text):
    # Find the start and end of the XML code block
    start_tag = "```xml"
    end_tag = "```"

    # Locate the positions of the start and end tags
    start_index = response_text.find(start_tag)
    end_index = response_text.find(end_tag, start_index + len(start_tag))

    # Extract the XML content between the tags
    if start_index != -1 and end_index != -1:
        # Adding len(start_tag) to skip the start tag and trimming any new lines or extra spaces
        xml_content = response_text[start_index + len(start_tag):end_index].strip()
        return xml_content
    else:
        return None


def save_to_drawio_file(xml_content, filename="diagram_Created.drawio"):
    """Saves the given XML content to a .drawio file in the 'data' directory."""
    if not xml_content:
        print("No XML content to save.")
        return

    # Create 'data' directory if it does not exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, filename)

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(xml_content)

    print(f"File saved at: {file_path}")