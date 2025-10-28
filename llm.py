from openai import OpenAI
import time

client = OpenAI()

start_time = time.time()
print("Hello")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "What is 3 + 3?"}
    ]
)

print(response.choices[0].message.content)
end_time = time.time()
print(end_time - start_time)
