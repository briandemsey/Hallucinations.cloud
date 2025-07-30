print("Script is running!")

import os
print("OS imported")

from dotenv import load_dotenv
print("dotenv imported")

load_dotenv()
print("dotenv loaded")

key = os.getenv("PERPLEXITY_API_KEY")
print(f"Key exists: {bool(key)}")

if key:
    print(f"First 5 chars: {key[:5]}")
else:
    print("No key found")

print("Script finished!")