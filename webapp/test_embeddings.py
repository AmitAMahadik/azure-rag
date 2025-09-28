# test_embeddings.py
import os
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

emb = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_API_VERSION","2024-10-21"),
    model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),  # embeddings deployment name (not base model)
)

print("Embedding dim:", len(emb.embed_query("hello world")))

