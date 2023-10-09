
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=0.1, max=0.2), stop=stop_after_attempt(10))
def call_gpt(chatgpt_messages, model="gpt-3.5-turbo", processor=None, temp_gpt=0.0):
#     import pdb; pdb.set_trace()
#     print(f"Start here: {chatgpt_messages}\n\n")
    prompt = ""
    for chat in chatgpt_messages:
        if chat['role'] == 'system':
            prompt += 'SYSTEM: \n\n'
            prompt += (chat['content']+'\n\n')
        elif chat['role'] == 'user':
            prompt += 'USER: \n\n'
            prompt += (chat['content']+'\n\n')
        else:
            prompt += 'ASSISTANT: \n\n'
            prompt += (chat['content']+'\n\n')
#     print(prompt)
    inputs = processor(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=256)
    reply = processor.decode(output[0], skip_special_tokens=True)
    total_tokens = 4096
#     print(reply)
#     print("End here:\n\n")
    
#     response = openai.ChatCompletion.create(model=model, messages=chatgpt_messages, temperature=temp_gpt, max_tokens=512)
#     reply = response['choices'][0]['message']['content']
#     total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

