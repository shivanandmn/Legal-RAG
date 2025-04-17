from llama_index.llms.openai import OpenAI
from llama_index.llms.vertex import Vertex
from google.oauth2 import service_account
from dotenv import load_dotenv, find_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel

_ = load_dotenv(find_dotenv())


filename = "creds\caselaws-421916-7be6b3326483.json"
credentials: service_account.Credentials = (
    service_account.Credentials.from_service_account_file(filename)
)
vertexai.init(
    project=credentials.project_id, location="us-central1", credentials=credentials
)


def get_gemini_model(model="gemini-1.5-flash", system_prompt=""):
    model = Vertex(
        model=model,
        project=credentials.project_id,  # "caselaws-421916",#semiotic-summer-423513-s2
        credentials=credentials,
        system_prompt=system_prompt,
        max_tokens=8_000,
    )

    # model._chat_client.start_chat(response_validation=False)
    # model._client.start_chat(response_validation=False)
    return model


def get_openai_model(model="gpt-3.5-turbo", system_prompt=""):
    return OpenAI(
        model=model,
        max_tokens=4_096,
        temperature=0.7,
        system_prompt=system_prompt,
    )


gemini_model = GenerativeModel("gemini-1.5-flash-001")


def get_gemini_token_count(text: str) -> int:
    response = gemini_model.count_tokens(text)
    return response.total_tokens
