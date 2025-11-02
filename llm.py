from openai import OpenAI

def llm_response(args, conversation):


    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation
    )

    print("LLM: ", response.choices[0].message.content)
    return response.choices[0].message.content
