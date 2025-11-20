from openai import OpenAI
from dotenv import load_dotenv

import os
load_dotenv(".env", override=True)

def llm_response(args, conversation):


    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=conversation
    )

    print("LLM: ", response.choices[0].message.content)
    return response.choices[0].message.content
