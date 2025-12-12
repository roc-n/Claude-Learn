import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
# model = "gpt-4o-mini"
model = "claude-3-5-haiku-20241022"

url = "https://api.laozhang.ai/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": api_key
}

def add_system_message(messages, content):
    messages.append({"role": "system", "content": content})

def add_user_message(messages, content):
    messages.append({"role": "user", "content": content})

def add_assistant_message(messages, content):
    messages.append({"role": "assistant", "content": content})

def chat(messages,temperature=None,stop_sequences=[]):
    params = {
        "model": model,
        "stream": False,
        "messages": messages,
        "stop": stop_sequences,
        "max_tokens": 1024,
    }
    if temperature:
        params["temperature"] = temperature

    response = requests.post(url, headers=headers, json=params)
    try: response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        print(f"Response content: {response.text}")
        raise

    answer = response.json()['choices'][0]['message']['content']
    return answer

import json

def generate_dataset():
    prompt = """
Generate an evaluation dataset for a prompt evaluation. The dataset will be used to evaluate prompts that generate Python, JSON, or Regex specifically for AWS-related tasks. Generate an array of JSON objects, each representing task that requires Python, JSON, or a Regex to complete.

Example output:
```json
[
  {
    "task": "Description of task",
    "type": "json|python|regex"
  },
  ...additional
]
```

* Focus on tasks that can be solved by writing a single Python function, a single JSON object, or a single regex
* Focus on tasks that do not require writing much code

Please generate 3 objects.
"""
    messages = []
    add_user_message(messages, prompt)
    add_assistant_message(messages, "```json")
    text = chat(messages, stop_sequences=["```"])
    
    print(text)
    return json.loads(text)

dataset = generate_dataset()