from rl_chain import ResponseValidator
from typing import Dict, Any

class MockValidator(ResponseValidator):
    def __init__(self):
        ...

    def grade_response(
        self, inputs: Dict[str, Any], llm_response: str, **kwargs
    ) -> float:
        return float(llm_response)