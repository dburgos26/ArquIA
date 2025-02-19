# GenAIProyect

This proyect was developed the possibility to apply generative artificial intelligence in the design process
of software architectures. This approach is built upon numerous gemini and ML models that have been trained
to fulfill specific tasks and work in colaboration to accomplish the objective of creating a software architecture
based on given requirements.

---

## Prerequisites

Ensure the following software and tools are installed before setting up the project:

- Python 3.12 or later
- Google Cloud Platform (GCP) account with the following services enabled:
  - AI Platform
  - Dialogflow CX
  - Cloud Storage
- A terminal or IDE to manage the project environment.

---

## Installation

1. **Clone the Repository**
   ```bash
   git clone <https://github.com/ThesisProyect/GenAIProyect.git>
   cd <GenAIProyect>

2. **Set up a Virtual Enviroment**s
    ```bash
    python3 -m venv env
    source env/bin/activate   # For Windows, use `env\\Scripts\\activate`

3. **Install Dependencies** Create a requirements.txt file with the following content:
    ```bash
    langchain-google-vertexai
    langchain-core
    langgraph
    google-cloud-aiplatform
    google-cloud-storage
    google-cloud-dialogflow-cx
    langchain-community
    langchain-openai
    langchain-text-splitters
    typing-extensions
    pydantic
    ```
    Then run
    
    ```bash
    pip install -r requirements.txt

4. **Set Up Google Cloud Credentials** To access GCP services like AI Platform, Dialogflow CX, and Cloud Storage, you need to configure credentials:
    - Create a service account in the Google Cloud Console.
    - Assign necessary roles (e.g., Vertex AI Admin, Dialogflow CX Agent, Storage Admin).
    - Download the service account key file (.json).
    - Set the enviroment variable:
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"


