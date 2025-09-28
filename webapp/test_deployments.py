#!/usr/bin/env python3

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_azure_deployments():
    print("Testing Azure OpenAI deployments...")
    
    # Print environment variables (masked)
    print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"AZURE_OPENAI_CHAT_ENDPOINT: {os.getenv('AZURE_OPENAI_CHAT_ENDPOINT')}")
    print(f"AZURE_OPENAI_API_KEY: {os.getenv('AZURE_OPENAI_API_KEY')[:10]}..." if os.getenv('AZURE_OPENAI_API_KEY') else None)
    print(f"AZURE_OPENAI_CHAT_DEPLOYMENT: {os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT')}")
    print(f"AZURE_OPENAI_EMBEDDING_DEPLOYMENT: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')}")
    print(f"AZURE_OPENAI_API_VERSION: {os.getenv('AZURE_OPENAI_API_VERSION')}")
    
    try:
        # Initialize client for chat
        chat_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
        )
        
        print("✅ Azure OpenAI chat client initialized successfully")
        
        # Test the chat deployment
        print(f"\nTesting chat deployment: {os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT')}")
        response = chat_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, can you recommend a good red wine?"}
            ],
            max_tokens=50
        )
        
        print("✅ Chat completion successful!")
        print(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Chat Error: {e}")
        
    # Test embeddings deployment
    try:
        print(f"\nTesting embeddings deployment: {os.getenv('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')}")
        embedding_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION", "2024-10-21"),
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        )
        
        response = embedding_client.embeddings.create(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            input="test text for embeddings"
        )
        
        print("✅ Embeddings successful!")
        print(f"Embedding dimensions: {len(response.data[0].embedding)}")
        
    except Exception as e:
        print(f"❌ Embeddings Error: {e}")

if __name__ == "__main__":
    test_azure_deployments()