#!/usr/bin/env python3

import os
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_full_rag():
    print("Testing complete RAG pipeline...")
    
    try:
        # Initialize Azure OpenAI client for chat
        endpoint = os.getenv("AZURE_OPENAI_CHAT_ENDPOINT")
        deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "demo-gpt-4.1-nano")
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        print("✅ Chat client initialized")
        
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        )
        print("✅ Embeddings initialized")
        
        # Connect to Azure Cognitive Search
        acs = AzureSearch(
            azure_search_endpoint=os.getenv('SEARCH_SERVICE_NAME'),
            azure_search_key=os.getenv('SEARCH_API_KEY'),
            index_name=os.getenv('SEARCH_INDEX_NAME'),
            embedding_function=embeddings.embed_query
        )
        print("✅ Azure Search connected")
        
        # Test search function
        def search(query):
            docs = acs.similarity_search_with_relevance_scores(
                query=query,
                k=5,
            )
            result = docs[0][0].page_content
            print(f"Search result: {result[:100]}...")
            return result
        
        # Test assistant function
        def assistant(query, context):
            messages = [
                {"role": "system", "content": "Assistant is a chatbot that helps you find the best wine for your taste."},
                {"role": "user", "content": query},
                {"role": "assistant", "content": context}
            ]
            
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
            )
            return response.choices[0].message.content
        
        # Test the complete pipeline
        test_query = "What is a good Cabernet Sauvignon?"
        print(f"\nTesting query: {test_query}")
        
        search_result = search(test_query)
        print("✅ Search completed")
        
        chat_response = assistant(test_query, search_result)
        print("✅ Chat completion successful")
        
        print(f"\nFinal Response:\n{chat_response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_rag()