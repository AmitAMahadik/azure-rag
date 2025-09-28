#!/usr/bin/env python3
"""
Test script to verify .env file loading and environment variables
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("🔍 Environment Variables Test")
print("=" * 50)

# Test all relevant environment variables
env_vars = [
    "OPENAI_API_BASE",
    "OPENAI_API_KEY", 
    "OPENAI_DEPLOYMENT",
    "SEARCH_SERVICE_NAME",
    "SEARCH_API_KEY",
    "SEARCH_INDEX_NAME"
]

for var in env_vars:
    value = os.getenv(var)
    if value:
        # Mask sensitive values
        if "KEY" in var:
            masked_value = f"***{value[-4:]}" if len(value) > 4 else "***"
            print(f"✅ {var}: {masked_value}")
        else:
            print(f"✅ {var}: {value}")
    else:
        print(f"❌ {var}: Not set")

print("=" * 50)
print("🎯 Summary:")
missing_vars = [var for var in env_vars if not os.getenv(var)]
if missing_vars:
    print(f"❌ Missing variables: {', '.join(missing_vars)}")
else:
    print("✅ All required environment variables are set!")

print("\n🔧 Next steps:")
if missing_vars:
    print("1. Add missing variables to your .env file")
else:
    print("1. Environment variables are loaded correctly")
    print("2. The authentication error suggests the API key or endpoint may be incorrect")
    print("3. Verify your Azure OpenAI resource is active and the endpoint/key are valid")