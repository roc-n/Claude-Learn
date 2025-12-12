import os
import json
from dotenv import load_dotenv
from openai import OpenAI


# Client Initialization and helper functions
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
# client = Anthropic()
client = OpenAI(
    api_key=api_key,
    base_url="https://api.laozhang.ai/v1"
)
# model = "claude-sonnet-4-20250514"
model = "gpt-4o-mini"


def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_system_message(messages, text):
    system_message = {"role": "system", "content": text}
    messages.append(system_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)


def chat(messages, temperature=1.0, stop_sequences=[]):
    params = {
        "model": model,
        "max_tokens": 1000,
        "messages": messages,
        "temperature": temperature,
        # "stop": stop_sequences,
    }

    response = client.chat.completions.create(**params,stop=stop_sequences)

    print(response)
    return response.choices[0].message.content


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