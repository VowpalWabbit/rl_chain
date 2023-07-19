from response_checker import ResponseChecker
from typing import Dict, Any

class MockChecker(ResponseChecker):
    def __init__(self):
        ...

    def grade_response(
        self, inputs: Dict[str, Any], llm_response: str, **kwargs
    ) -> float:
        return float(llm_response)