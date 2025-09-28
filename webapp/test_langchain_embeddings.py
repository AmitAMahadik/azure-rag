#!/usr/bin/env python3

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_embeddings_initialization():
    print("Testing LangChain Azure OpenAI Embeddings initialization...")
    
    try:
        from langchain_openai import AzureOpenAIEmbeddings
        
        # Test with azure_deployment parameter (should work with newer versions)
        try:
            embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            )
            print("✅ AzureOpenAIEmbeddings initialized with azure_deployment parameter")
            
            # Test the embedding function
            test_text = "This is a test embedding"
            embedding = embeddings.embed_query(test_text)
            print(f"✅ Embedding generated successfully, dimension: {len(embedding)}")
            
        except Exception as e:
            print(f"❌ Error with azure_deployment parameter: {e}")
            
            # Try with model parameter (fallback for older versions)
            try:
                embeddings = AzureOpenAIEmbeddings(
                    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
                    model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                )
                print("✅ AzureOpenAIEmbeddings initialized with model parameter (fallback)")
                
                # Test the embedding function
                test_text = "This is a test embedding"
                embedding = embeddings.embed_query(test_text)
                print(f"✅ Embedding generated successfully, dimension: {len(embedding)}")
                
            except Exception as e2:
                print(f"❌ Error with model parameter: {e2}")
                
    except ImportError as e:
        print(f"❌ Import error: {e}")

if __name__ == "__main__":
    test_embeddings_initialization()