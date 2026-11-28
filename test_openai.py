from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, AuthenticationError
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=0
)

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Say hello"}
        ],
        max_tokens=20
    )

    print("SUCCESS:", response.choices[0].message.content)

except RateLimitError as e:
    print("OPENAI 429 ERROR")
    print("Details:", e)

except AuthenticationError:
    print("Invalid OpenAI API key")