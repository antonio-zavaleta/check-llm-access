from openai import OpenAI
from anthropic import Anthropic 
from llamaapi import LlamaAPI
from mistralai import Mistral
import google.generativeai as genai
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv('llm_api_params.env')

logger = logging.getLogger(__name__)

class ApiConnInfoCarrier(ABC):
    """
    Abstract base class that carries API connection information.
    Attributes:
        _conn_params (dict): A dictionary to store connection parameters.
    """
    def __init__(self):
        """
        Initializes the ApiConnInfoCarrier with an empty dictionary for connection parameters.
        """
        self._conn_params = {}

    @property
    def conn_params(self):
        """
        Returns the connection parameters.
        """
        return self._conn_params
    
class OpenAiConnInfoCarrier(ApiConnInfoCarrier):
    """
    OpenAiConnInfoCarrier is a class that carries connection information for OpenAI API.
    Attributes:
        openai_api_key (str): The API key for accessing OpenAI services.
        model (str): The model to be used for OpenAI API requests.
        role (str): The role associated with the API connection.
    """
    def __init__(self, openai_api_key: str, model: str, role: str):
        super().__init__()
        self._conn_params['openai_api_key'] = openai_api_key
        self._conn_params['model'] = model
        self._conn_params['role'] = role
        

class GeminiConnInfoCarrier(ApiConnInfoCarrier):
    """
    A class to carry connection information for the Gemini API.
    Attributes:
        gemini_api_key (str): The API key for accessing the Gemini API.
        model_name (str): The name of the model to be used with the Gemini API.
    """
    def __init__(self, gemini_api_key: str, model_name: str):
        super().__init__()
        self._conn_params['gemini_api_key'] = gemini_api_key
        self._conn_params['model_name'] = model_name


class ClaudeInfoCarrier(ApiConnInfoCarrier):
    """
    ClaudeInfoCarrier is a class that carries connection information for Anthropic's Claude API.
    Attributes:
        claude_api_key (str): The API key for accessing Claude services.
        model (str): The model to be used for Claude API requests.
    """
    def __init__(self, claude_api_key: str,
                 model: str,
                 role: str,
                 max_tokens: int
                 ):
        super().__init__()
        self._conn_params['claude_api_key'] = claude_api_key
        self._conn_params['model'] = model
        self._conn_params['role'] = role
        self._conn_params['max_tokens'] = max_tokens

class LlamaInfoCarrier(ApiConnInfoCarrier):
    """
    LLamaInfoCarrier is a class that carries connection information for LLama API.
    Attributes:
        llama_api_key (str): The API key for accessing LLama services.
        model (str): The model to be used for LLama API requests.
        role (str): The role associated with the API connection.
        max_tokens (int): The maximum number of tokens for the API request.
    """
    def __init__(self, llama_api_key: str, hostname: str, model: str, role: str, max_tokens: int):
        super().__init__()
        self._conn_params['llama_api_key'] = llama_api_key
        self._conn_params['model'] = model
        self._conn_params['role'] = role
        self._conn_params['max_tokens'] = max_tokens
        self._conn_params['hostname'] = hostname

class MistralInfoCarrier(ApiConnInfoCarrier):
    """
    MistralInfoCarrier is a class that carries connection information for Mistral API.
    Attributes:
        mistral_api_key (str): The API key for accessing Mistral services.
        model (str): The model to be used for Mistral API requests.
        role (str): The role associated with the API connection.
        max_tokens (int): The maximum number of tokens for the API request.
    """
    def __init__(self, mistral_api_key: str, model: str, role: str):
        super().__init__()
        self._conn_params['mistral_api_key'] = mistral_api_key
        self._conn_params['model'] = model
        self._conn_params['role'] = role

class LlmQuerier(ABC):
    """
    Abstract base class for querying language models (LLMs).
    Attributes:
        api_info_carrier (ApiConnInfoCarrier): An object containing API connection information.
    Methods:
        get_query_results(prompt: str):
            Abstract method to get query results from the LLM.
        get_lm_conn_obj(subclass_name: str) -> 'LlmQuerier':
            Static method to get an instance of a subclass based on the provided subclass name.
    """
    def __init__(self, api_info_carrier: ApiConnInfoCarrier):
        self.api_info_carrier = api_info_carrier

    @abstractmethod
    def get_query_results(self, prompt: str):
        pass

    @staticmethod
    def get_lm_conn_obj(subclass_name: str)->'LlmQuerier':
        subclasses = {cls.__name__: cls for cls in LlmQuerier.__subclasses__()}
        if subclass_name in subclasses:
            if subclass_name == "OpenAiQuerier":
                kwargs = {
                    "openai_api_key":os.getenv("OPENAI_API_KEY"),
                    "model":os.getenv("OPENAI_MODEL"),
                    "role":os.getenv("OPENAI_ROLE")
                }
                conn_info_ds = OpenAiConnInfoCarrier(**kwargs)
                llm_query_obj = OpenAiQuerier(conn_info_ds)
            elif subclass_name == "GeminiQuerier":
                kwargs = {
                    "gemini_api_key":os.getenv("GEMINI_API_KEY"),
                    "model_name":os.getenv("GEMINI_MODEL_NAME")
                }
                conn_info_ds = GeminiConnInfoCarrier(**kwargs)
                llm_query_obj = GeminiQuerier(conn_info_ds)
            
            elif subclass_name == "ClaudeQuerier":
                kwargs = {
                    "claude_api_key":os.getenv("CLAUDE_API_KEY"),
                    "model":os.getenv("CLAUDE_MODEL"),
                    "role":os.getenv("CLAUDE_ROLE"),
                    "max_tokens":int(os.getenv("CLAUDE_MAX_TOKENS"))
                }
                conn_info_ds = ClaudeInfoCarrier(**kwargs)
                llm_query_obj = ClaudeQuerier(conn_info_ds)
        
            elif subclass_name == "LlamaQuerier":
                kwargs = {
                    "llama_api_key":os.getenv("LLAMA_API_KEY"),
                    "model":os.getenv("LLAMA_MODEL"),
                    "role":os.getenv("LLAMA_ROLE"),
                    "max_tokens":int(os.getenv("LLAMA_MAX_TOKENS")),
                    "hostname":os.getenv("LLAMA_HOSTNAME")
                }
                conn_info_ds = LlamaInfoCarrier(**kwargs)
                llm_query_obj = LlamaQuerier(conn_info_ds)
                
            elif subclass_name == "MistralQuerier":
                kwargs = {
                    "mistral_api_key":os.getenv("MISTRAL_API_KEY"),
                    "model":os.getenv("MISTRAL_MODEL"),
                    "role":os.getenv("MISTRAL_ROLE")
                }
                conn_info_ds = MistralInfoCarrier(**kwargs)
                llm_query_obj = MistralQuerier(conn_info_ds)
                
            return llm_query_obj
        else:
            raise ValueError(f"Unknown subclass name: {subclass_name}")


class OpenAiQuerier(LlmQuerier):
    """A class to interact with the OpenAI API to send prompts and retrieve responses.
    Attributes:
        api_info_carrier (OpenAiConnInfoCarrier): An object containing connection information for the OpenAI API.
    """
    def __init__(self, api_info_carrier: OpenAiConnInfoCarrier=None):
        
        if not api_info_carrier:
            api_info_carrier = OpenAiConnInfoCarrier(self.__class__.__name__)        
        super().__init__(api_info_carrier)

    def get_query_results(self, prompt: str):
        """
        Sends a prompt to the OpenAI ChatGPT model and returns the generated response.
        Args:
            prompt (str): The input text prompt to be sent to the ChatGPT model.
        Returns:
            str: The response generated by the ChatGPT model based on the input prompt.
        """
        openai_api_key = self.api_info_carrier.conn_params['openai_api_key']
        role = self.api_info_carrier.conn_params['role']
        model = self.api_info_carrier.conn_params['model']        
        client = OpenAI(api_key=openai_api_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": role,"content": prompt},      ],
            model=model,
        )
        return chat_completion.choices[0].message.content

class GeminiQuerier(LlmQuerier):
    """
    A class to query the Gemini generative model.
    Attributes:
        api_info_carrier (GeminiConnInfoCarrier): An object containing API connection information.
    Methods:
        get_query_results(query: str) -> str:
            Queries the Gemini generative model with the provided query and returns the response text.
    """ 
    def __init__(self, api_info_carrier: GeminiConnInfoCarrier):
        if not api_info_carrier:
            api_info_carrier = GeminiConnInfoCarrier(self.__class__.__name__)

        super().__init__(api_info_carrier)

    def get_query_results(self, query: str):
        """
        Queries the Gemini generative model with the provided query.

        Args:
            gemini_api_key (str): The API key for the Gemini API.
            model_name (str): The name of the Gemini model to query.
            query (str): The query to send to the Gemini model.

        Returns:
            requests.Response: The response object from the Gemini API.
        """
        gemini_api_key = self.api_info_carrier.conn_params['gemini_api_key']
        model_name = self.api_info_carrier.conn_params['model_name']
        
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)
        gemini_response = model.generate_content(query)
        logger.debug(f"Gemini Suggested Topics: {gemini_response.text}")

        return gemini_response.text

class ClaudeQuerier(LlmQuerier):
    """
    A class to query the Claude generative model.
    Attributes:
        api_info_carrier (ClaudeInfoCarrier): An object containing API connection information.
    Methods:
        get_query_results(prompt: str) -> str:
            Queries the Claude generative model with the provided prompt and returns the response text.
    """
    def __init__(self, api_info_carrier: ClaudeInfoCarrier):
        if not api_info_carrier:
            api_info_carrier = ClaudeInfoCarrier(self.__class__.__name__)
        super().__init__(api_info_carrier)

    def get_query_results(self, prompt: str):
        """
        Queries the Claude generative model with the provided prompt.

        Args:
            prompt (str): The input text prompt to be sent to the Claude model.

        Returns:
            str: The response generated by the Claude model based on the input prompt.
        """
        claude_api_key = self.api_info_carrier.conn_params['claude_api_key']
        model = self.api_info_carrier.conn_params['model']
        role = self.api_info_carrier.conn_params['role']
        max_tokens = self.api_info_carrier.conn_params['max_tokens']
        
        # Create model client
        client = Anthropic(api_key=claude_api_key,)
        
        # Query the Claude model via clien
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": role, "content": prompt}
            ]
        )
        
        return message.content[0].text
    
class LlamaQuerier(LlmQuerier):
    """
    A class to query the Llama generative model.
    Attributes:
        api_info_carrier (LlamaInfoCarrier): An object containing API connection information.
    Methods:
        get_query_results(prompt: str) -> str:
            Queries the Llama generative model with the provided prompt and returns the response text.
    """
    def __init__(self, api_info_carrier: LlamaInfoCarrier):
        if not api_info_carrier:
            api_info_carrier = LlamaInfoCarrier(self.__class__.__name__)
        super().__init__(api_info_carrier)

    def get_query_results(self, prompt: str):
        """
        Queries the Llama generative model with the provided prompt.

        Args:
            prompt (str): The input text prompt to be sent to the Llama model.

        Returns:
            str: The response generated by the Llama model based on the input prompt.
        """
        llama_api_key = self.api_info_carrier.conn_params['llama_api_key']
        model = self.api_info_carrier.conn_params['model']
        role = self.api_info_carrier.conn_params['role']
        max_tokens = self.api_info_carrier.conn_params['max_tokens']
        hostname = self.api_info_carrier.conn_params['hostname']
        # Create model client
        #client = LlamaAPI(api_token=llama_api_key, hostname=hostname)
        llama = LlamaAPI(api_token=llama_api_key)
        
        api_request_json = {
        "model": model,
        "messages": [{"role": role, "content": prompt},],
        "max_tokens": max_tokens
        }
        
        response = llama.run(api_request_json).json()
        
        return response['choices'][0]['message']['content']
    
class MistralQuerier(LlmQuerier):
    """
    A class to query the Mistral generative model.
    Attributes:
        api_info_carrier (MistralInfoCarrier): An object containing API connection information.
    Methods:
        get_query_results(prompt: str) -> str:
            Queries the Mistral generative model with the provided prompt and returns the response text.
    """
    def __init__(self, api_info_carrier: MistralInfoCarrier):
        if not api_info_carrier:
            api_info_carrier = MistralInfoCarrier(self.__class__.__name__)
        super().__init__(api_info_carrier)

    def get_query_results(self, prompt: str):
        """
        Queries the Mistral generative model with the provided prompt.
        Args:
            prompt (str): The input text prompt to be sent to the Mistral model.

        Returns:
            str: The response generated by the Mistral model based on the input prompt.
        """
        mistral_api_key = self.api_info_carrier.conn_params['mistral_api_key']
        model = self.api_info_carrier.conn_params['model']
        role = self.api_info_carrier.conn_params['role']
        
        # Create model client
        client = Mistral(api_key=mistral_api_key)
        
        # Query the Mistral model via client
        chat_response = client.chat.complete(
            model= model,
            messages = [
                {
                    "role": role,
                    "content": prompt,
                },
            ]
        )
        
        return chat_response.choices[0].message.content
        