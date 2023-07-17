from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


class ResponseChecker(ABC):
    """Abstract method to grade the chosen action or the response of the llm"""

    @abstractmethod
    def grade_response(
        self, inputs: Dict[str, Any], llm_response: str, **kwargs
    ) -> float:
        pass


class LLMResponseCheckerForCB(ResponseChecker):
    llm_chain: LLMChain
    prompt: PromptTemplate

    def __init__(self, llm, prompt=None):
        if prompt:
            self.prompt = prompt
        else:
            template = "PLEASE RESPOND ONLY WITH A SIGNLE FLOAT AND NO OTHER TEXT EXPLANATION\n You are a VERY VERY strict judge that is called on to rank a response based on given criteria.\
                You must respond with your ranking by providing a single float within the range [-1, 1], -1 being very bad response and 1 being very good response."
            system_message_prompt = SystemMessagePromptTemplate.from_template(template)
            human_template = "Given this context {context} as the most important attribute, rank how good or bad this text selection is: {action}."
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                human_template
            )

            chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt]
            )
            self.prompt = chat_prompt

        self.llm_chain = LLMChain(llm=llm, prompt=self.prompt)

    def grade_response(
        self, inputs: Dict[str, Any], llm_response: str, **kwargs
    ) -> float:
        
        if "chosen_action" not in kwargs:
            raise ValueError("The chosen action is not provided to the LLM response Checker, please provide the chosen action")

        chosen_action = kwargs["chosen_action"]
        
        ranking = self.llm_chain.predict(**inputs, action=chosen_action)
        ranking = ranking.strip()
        try:
            resp = float(ranking)
            return resp
        except Exception:
            raise RuntimeError(
                "The llm did not manage to rank the response as expected, there is always the option to try again"
            )
