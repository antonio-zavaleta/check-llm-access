import unittest
import os
from src.apiconn import OpenAiConnInfoCarrier
from src.apiconn import GeminiConnInfoCarrier
from src.apiconn import ClaudeInfoCarrier
from src.apiconn import LlmQuerier
from src.apiconn import LlamaInfoCarrier

class TestOpenAiConnInfoCarrier(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.model = "test_model"
        self.role = "system"
        self.conn_info_carrier = OpenAiConnInfoCarrier(self.api_key, self.model, self.role)

    def test_initialization(self):
        self.assertEqual(self.conn_info_carrier._conn_params['openai_api_key'],
                         self.api_key)
        self.assertEqual(self.conn_info_carrier._conn_params['model'],
                         self.model)
        self.assertEqual(self.conn_info_carrier._conn_params['role'],
                         self.role)


class TestGeminiConnInfoCarrier(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_gemini_api_key"
        self.model = "test_gemini_model"
        self.conn_info_carrier = GeminiConnInfoCarrier(self.api_key, self.model)

    def test_initialization(self):
        self.assertEqual(self.conn_info_carrier._conn_params['gemini_api_key'],
                            self.api_key)
        self.assertEqual(self.conn_info_carrier._conn_params['model_name'],
                            self.model)

class TestClaudeInfoCarrier(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_claude_api_key"
        self.model = "test_claude_model"
        self.role = "user"
        self.max_tokens = 1024
        self.conn_info_carrier = ClaudeInfoCarrier(self.api_key, self.model, self.role, self.max_tokens)

    def test_initialization(self):
        self.assertEqual(self.conn_info_carrier._conn_params['claude_api_key'],
                            self.api_key)
        self.assertEqual(self.conn_info_carrier._conn_params['model'],
                            self.model)
        self.assertEqual(self.conn_info_carrier._conn_params['role'],
                            self.role)
        self.assertEqual(self.conn_info_carrier._conn_params['max_tokens'],
                            self.max_tokens)
        
class TestLlamaInfoCarrier(unittest.TestCase):

    def setUp(self):
        self.llama_api_key = "test_llama_api_key"
        self.model = "test_model"
        self.role = "test_role"
        self.max_tokens = 1000
        self.base_url = "https://llama.com"
        self.llama_info_carrier = LlamaInfoCarrier(
            llama_api_key=self.llama_api_key,
            model=self.model,
            role=self.role,
            base_url=self.base_url,
            max_tokens=self.max_tokens
        )

    def test_conn_params(self):
        self.assertEqual(self.llama_info_carrier.conn_params['llama_api_key'], self.llama_api_key)
        self.assertEqual(self.llama_info_carrier.conn_params['model'], self.model)
        self.assertEqual(self.llama_info_carrier.conn_params['role'], self.role)
        self.assertEqual(self.llama_info_carrier.conn_params['max_tokens'], self.max_tokens)

class TestLlmQuerier(unittest.TestCase):
    def test_get_lm_conn_obj_openai(self):
        os.environ['OPENAI_API_KEY'] = 'test_openai_api_key'
        os.environ['OPENAI_MODEL'] = 'test_openai_model'
        os.environ['OPENAI_ROLE'] = 'test_openai_role'
        lm_query_obj = LlmQuerier.get_lm_conn_obj('OpenAiQuerier')
        self.assertIsInstance(lm_query_obj.api_info_carrier, OpenAiConnInfoCarrier)
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['openai_api_key'], 'test_openai_api_key')
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['model'], 'test_openai_model')
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['role'], 'test_openai_role')

    def test_get_lm_conn_obj_gemini(self):
        os.environ['GEMINI_API_KEY'] = 'test_gemini_api_key'
        os.environ['GEMINI_MODEL_NAME'] = 'test_gemini_model'
        lm_query_obj = LlmQuerier.get_lm_conn_obj('GeminiQuerier')
        self.assertIsInstance(lm_query_obj.api_info_carrier, GeminiConnInfoCarrier)
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['gemini_api_key'], 'test_gemini_api_key')
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['model_name'], 'test_gemini_model')

    def test_get_lm_conn_obj_claude(self):
        os.environ['CLAUDE_API_KEY'] = 'test_claude_api_key'
        os.environ['CLAUDE_MODEL'] = 'test_claude_model'
        os.environ['CLAUDE_ROLE'] = 'test_claude_role'
        os.environ['CLAUDE_MAX_TOKENS'] = '1024'
        lm_query_obj = LlmQuerier.get_lm_conn_obj('ClaudeQuerier')
        self.assertIsInstance(lm_query_obj.api_info_carrier, ClaudeInfoCarrier)
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['claude_api_key'], 'test_claude_api_key')
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['model'], 'test_claude_model')
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['role'], 'test_claude_role')
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['max_tokens'], 1024)
    
    def test_get_lm_conn_obj_llama(self):
        os.environ['LLAMA_API_KEY'] = 'test_llama_api_key'
        os.environ['LLAMA_MODEL'] = 'test_llama_model'
        os.environ['LLAMA_ROLE'] = 'test_llama_role'
        os.environ['LLAMA_MAX_TOKENS'] = '1000'
        os.environ['LLAMA_BASE_URL'] = 'https://llama.com'
        lm_query_obj = LlmQuerier.get_lm_conn_obj('LlamaQuerier')
        self.assertIsInstance(lm_query_obj.api_info_carrier, LlamaInfoCarrier)
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['llama_api_key'], 'test_llama_api_key')
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['model'], 'test_llama_model')
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['role'], 'test_llama_role')
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['base_url'], 'https://llama.com')
        self.assertEqual(lm_query_obj.api_info_carrier.conn_params['max_tokens'], 1000)
    
    def test_get_lm_conn_obj_invalid(self):
        with self.assertRaises(ValueError):
            LlmQuerier.get_lm_conn_obj('InvalidQuerier')

if __name__ == '__main__':
    unittest.main()