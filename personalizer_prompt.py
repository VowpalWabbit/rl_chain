# flake8: noqa
from __future__ import annotations

from langchain.prompts.prompt import PromptTemplate

_PROMPT_TEMPLATE = """You are given a set of action(s), here are their description(s): {actions}.

You have to embed these action(s) into the text where it makes sense. Put the action(s) after the words "Selections:" and end the actions with [end]. Here is the text: This weeks delicous selections tailored just for you based on your statements that you want {text_to_personalize}.

Don't leave any action(s) out.
"""


PROMPT = PromptTemplate(
    input_variables=["actions", "text_to_personalize"],
    template=_PROMPT_TEMPLATE,
)
