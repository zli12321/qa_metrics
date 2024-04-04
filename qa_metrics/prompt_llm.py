import os, requests
from typing import Union, List
import openai
from openai import OpenAI


'''
This file support calling OPENAI, Anthropic, and Cohere API nd get model prompt for text generation and completion
'''

class CloseLLM:
    def __init__(self):
        self.openai = None
        self.anthropic = None

    def set_openai_api_key(self, api_key):
        os.environ["OPENAI_API_KEY"] = api_key
        self.openai = OpenAI(api_key=api_key)  

    def set_anthropic_api_key(self, api_key):
        os.environ["ANTHROPIC_API_KEY"] = api_key
        api_key = os.environ["ANTHROPIC_API_KEY"]

    '''
    Calls openai gpt functions. Given a message, it returns the response from the model. All parameters are at OPENAI's default value unless you specify it.
    '''
    def prompt_gpt(self, prompt: str, 
                    model_engine: str='gpt-3.5-turbo', 
                    frequency_penalty: float = 0, 
                    logit_bias: dict = None, 
                    max_tokens: int = None,  # Assuming `inf` implies no limit, thus defaulting to `None`
                    n: int = 1, 
                    presence_penalty: float = 0, 
                    stop: Union[str, List[str]] = None,   # Type hint for either a string or a list
                    stream: bool = False, 
                    temperature: float = 0.7, 
                    top_p: float = 1, 
                    user: str = None):
        '''
        Function to interact with an AI model using given parameters.

        :param model: Model string, specifying which AI model to use.
        :param messages: Array of messages to be processed by the model.
        :param frequency_penalty: Adjusts frequency of repeated content in generations.
        :param logit_bias: Map of logit modifiers to adjust model's output.
        :param max_tokens: Maximum number of tokens to generate.
        :param n: Number of completions to generate.
        :param presence_penalty: Adjusts likelihood of new content in generations.
        :param stop: Sequence(s) at which text generation will be stopped.
        :param stream: Whether to stream the output.
        :param temperature: Controls randomness of generations.
        :param top_p: Controls diversity of generations via nucleus sampling.
        :param user: Optional user identifier.
        :return: The output from the model.
        # Get the OpenAI API key from the environment variable
        openai.api_key = os.environ["OPENAI_API_KEY"]
        '''

        try:
            # Send the request to the OpenAI API
            # response = client.completions.create(engine=model_engine,
            # prompt=prompt,
            # max_tokens=max_tokens,
            # temperature=temperature,
            # n=1,  # Number of responses to generate
            # stop=None)

            # # Extract the generated response from the API response
            # generated_text = response.choices[0].text.strip()

            # return generated_text

            completion = self.openai.chat.completions.create(
                model=model_engine,
                messages=[{"role": "user", "content": prompt}], 
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,  # Number of responses to generate
                stop=None
            )

            message_content = completion.choices[0].message.content

            return message_content
        except Exception as e:
            # message = [{ "role": "user", "content": prompt}]
            # answer = client.chat.completions.create(model = model_engine,
            # messages=message, 
            # frequency_penalty=frequency_penalty,
            # logit_bias=logit_bias,
            # max_tokens=max_tokens,
            # n=n,
            # presence_penalty=presence_penalty,
            # stop=stop,
            # temperature=temperature,
            # top_p=top_p,
            # stream=stream,
            # user=user)

            try:
                # return answer.choices[0].message.content
                completion = self.openai.completions.create(
                    model=model_engine,
                    prompt=prompt,
                    frequency_penalty=frequency_penalty,
                    max_tokens=max_tokens,
                    n=n,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream
                )

                return completion.choices[0].text
            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")
                return None

            except KeyError:
                print("OpenAI API key not found in the environment variables.")
                return None


    def prompt_claude(self, prompt, model_engine="claude-v1", anthropic_version="2023-06-01", max_tokens_to_sample=100, temperature=0.7):
        # Set up the API endpoint URL
        url = "https://api.anthropic.com/v1/complete"

        # Set up the request headers
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": anthropic_version
        }


        # print(conversation_history)
        conversation_history = "Human: " + prompt + "\nAssistant:"

        # Set up the request payload
        payload = {
            "prompt": conversation_history,
            "model": model_engine,
            "max_tokens_to_sample": max_tokens_to_sample,
            "temperature": temperature
        }

        try:
            # Send the request to the Anthropic API
            response = requests.post(url, json=payload, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data["completion"]
                return generated_text.strip()
            else:
                print(f"Anthropic API request failed with status code: {response.status_code}")
                print(f"Error message: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while calling the Anthropic API: {e}")
            return None

        except KeyError:
            print("Anthropic API key not found in the environment variables.")
            return None
