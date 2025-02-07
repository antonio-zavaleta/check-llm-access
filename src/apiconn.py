from abc import ABC, abstractmethod


class ApiConnInfoCarrier(ABC):
    def __init__(self):
        self._conn_params = {}

    @property
    def conn_params(self):
        return self._conn_params
    

class OpenAiConnInfoCarrier(ApiConnInfoCarrier):
    def __init__(self, openai_api_key: str, model: str, messages: list[dict]):
        super().__init__()
        self._conn_params['openai_api_key'] = openai_api_key
        self._conn_params['model'] = model
        self._conn_params['messages'] = messages
        
        
class GeminiConnInfoCarrier(ApiConnInfoCarrier):
    def __init__(self, gemini_api_key: str, model_name: str, content_type: str = "application/json"):
        super().__init__()
        self._conn_params['gemini_api_key'] = gemini_api_key
        self._conn_params['model_name'] = model_name
        self._conn_params['content_type'] = content_type