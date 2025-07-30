import os
import requests
from dotenv import load_dotenv

load_dotenv()
perplexity_key = os.getenv("PERPLEXITY_API_KEY")

# List of models to test
models_to_test = [
    "sonar",
    "sonar-small",
    "sonar-medium",
    "sonar-small-chat",
    "sonar-medium-chat",
    "sonar-small-online",
    "sonar-medium-online",
    "mixtral-8x7b-instruct",
    "llama-3-sonar-small-32k-online",
    "llama-3-sonar-large-32k-online",
    "llama-3.1-sonar-small-128k-online",
    "llama-3-70b-instruct",
    "llama-3-8b-instruct"
]

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {perplexity_key}"
}

print("Testing Perplexity models...\n")

for model in models_to_test:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        json=payload,
        headers=headers
    )
    
    if response.status_code == 200:
        print(f"✅ {model} - WORKS!")
    else:
        error_msg = "Unknown error"
        try:
            error_msg = response.json().get('error', {}).get('message', response.text[:50])
        except:
            pass
        print(f"❌ {model} - {error_msg}")