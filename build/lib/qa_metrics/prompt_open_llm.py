# Assume openai>=1.0.0
from openai import OpenAI


'''
Supports Various Open-Sources LLMs by calling API functions: 
# model="lizpreciatior/lzlv_70b_fp16_hf",
# model="meta-llama/Llama-2-70b-chat-hf",
model="meta-llama/Llama-2-7b-chat-hf",
model="meta-llama/Llama-2-13b-chat-hf",
# model="01-ai/Yi-34B-Chat",
# model="google/gemma-7b-it",
# model="llava-hf/llava-1.5-7b-hf",
# model="mistralai/Mixtral-8x7B-Instruct-v0.1"
See link for API key and models: https://deepinfra.com/models
'''

class OpenLLM:
    def __init__(self):
        self.openai = None

    def set_deepinfra_key(self, api_key):
        self.openai = OpenAI(
            api_key=api_key,
            base_url="https://api.deepinfra.com/v1/openai",
        )

    def prompt(self, message, model_engine="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0.7, max_tokens=100, top_p=0.5):
        chat_completion = self.openai.chat.completions.create(
        model = model_engine,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        # num_return_sequences=2,
        messages=[{"role": "user", "content": message}],
        )

        return chat_completion.choices[0].message.content
    
