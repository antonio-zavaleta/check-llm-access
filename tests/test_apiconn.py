import unittest
from src.apiconn import OpenAiConnInfoCarrier
from src.apiconn import GeminiConnInfoCarrier

class TestOpenAiConnInfoCarrier(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.model = "test_model"
        self.messages = [{"role": "system", "content": "Hello, World!"}]
        self.conn_info_carrier = OpenAiConnInfoCarrier(self.api_key, self.model, self.messages)

    def test_initialization(self):
        self.assertEqual(self.conn_info_carrier._conn_params['openai_api_key'],
                         self.api_key)
        self.assertEqual(self.conn_info_carrier._conn_params['model'],
                         self.model)
        self.assertEqual(self.conn_info_carrier._conn_params['messages'],
                         self.messages)


class TestGeminiConnInfoCarrier(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_gemini_api_key"
        self.model = "test_gemini_model"
        self.content_type = "application/json"
        self.conn_info_carrier = GeminiConnInfoCarrier(self.api_key, self.model, self.content_type)

    def test_initialization(self):
        self.assertEqual(self.conn_info_carrier._conn_params['gemini_api_key'],
                            self.api_key)
        self.assertEqual(self.conn_info_carrier._conn_params['model_name'],
                            self.model)
        self.assertEqual(self.conn_info_carrier._conn_params['content_type'],
                            self.content_type)

if __name__ == '__main__':
    unittest.main()