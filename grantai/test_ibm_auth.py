from dotenv import load_dotenv
import os
import requests

load_dotenv(override=True)

key = os.getenv("IBM_API_KEY", "").strip()

print("Key loaded:", bool(key))
print("Key length:", len(key))

response = requests.post(
    "https://iam.cloud.ibm.com/identity/token",
    data={
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": key,
    },
    timeout=30,
)

print("HTTP status:", response.status_code)
print("Response:", response.text)