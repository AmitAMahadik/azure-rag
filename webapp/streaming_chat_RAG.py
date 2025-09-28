# streaming_chat_completion.py
import os
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

#endpoint = "https://oai-demo-search.openai.azure.com/"
endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
model_name = os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME", "gpt-4.1-nano")
deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "demo-gpt-4.1-nano")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")   

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
)

# Connect to Azure Cognitive Search
# Note: SEARCH_SERVICE_NAME contains full URL, so we use it as azure_search_endpoint
acs = AzureSearch(azure_search_endpoint=os.getenv('SEARCH_SERVICE_NAME'),
                 azure_search_key=os.getenv('SEARCH_API_KEY'),
                 index_name=os.getenv('SEARCH_INDEX_NAME'),
                 embedding_function=embeddings.embed_query)

class Body(BaseModel):
    query: str

@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')

def ask(body: Body):
    """
    Use the query parameter to interact with the Azure OpenAI Service
    using the Azure Cognitive Search API for Retrieval Augmented Generation.
    """
    search_result = search(body.query)
    chat_bot_response = assistant(body.query, search_result)
    return {'response': chat_bot_response}



def search(query):
    """
    Send the query to Azure Cognitive Search and return the top result
    """
    docs = acs.similarity_search_with_relevance_scores(
        query=query,
        k=5,
    )
    result = docs[0][0].page_content
    print(result)
    return result


def assistant(query, context):
    messages=[
        # Set the system characteristics for this chat bot
        {"role": "system", "content": "Assistant is a chatbot that helps you find the best wine for your taste."},

        # Set the query so that the chatbot can respond to it
        {"role": "user", "content": query},

        # Add the context from the vector search results so that the chatbot can use
        # it as part of the response for an augmented context
        {"role": "assistant", "content": context}
    ]


    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
    )
    return response.choices[0].message.content

'''
response = client.chat.completions.create(
    stream=True,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=13107,
    temperature=1.0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    model=deployment,
)

for update in response:
    if update.choices:
        print(update.choices[0].delta.content or "", end="")

del acs  # ensure destructor runs before interpreter shutdown

client.close()

'''