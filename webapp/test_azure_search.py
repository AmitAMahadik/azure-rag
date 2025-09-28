#!/usr/bin/env python3

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_azure_search():
    print("Testing Azure Cognitive Search connection...")
    
    try:
        from langchain_openai import AzureOpenAIEmbeddings
        from langchain_community.vectorstores import AzureSearch
        
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        )
        print("✅ Embeddings initialized")
        
        # Initialize Azure Search
        acs = AzureSearch(
            azure_search_endpoint=os.getenv('SEARCH_SERVICE_NAME'),
            azure_search_key=os.getenv('SEARCH_API_KEY'),
            index_name=os.getenv('SEARCH_INDEX_NAME'),
            embedding_function=embeddings.embed_query
        )
        print("✅ Azure Search initialized")
        
        # Test a simple search
        print("\nTesting search functionality...")
        test_query = "red wine"
        docs = acs.similarity_search_with_relevance_scores(
            query=test_query,
            k=3,
        )
        
        print(f"✅ Search completed! Found {len(docs)} results")
        if docs:
            print(f"Top result score: {docs[0][1]:.4f}")
            print(f"Top result preview: {docs[0][0].page_content[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_azure_search()