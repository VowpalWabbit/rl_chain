import sys

sys.path.append("..")

import rl_chain.pick_best_chain as pick_best_chain
from test_utils import MockEncoder
import pytest
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import FakeListChatModel


def setup():
    _PROMPT_TEMPLATE = """This is a dummy prompt that will be ignored by the fake llm"""
    PROMPT = PromptTemplate(input_variables=[], template=_PROMPT_TEMPLATE)

    llm = FakeListChatModel(responses=["hey"])
    return llm, PROMPT


def test_multiple_ToSelectFrom_throws():
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(llm=llm, prompt=PROMPT)
    actions = ["0", "1", "2"]
    with pytest.raises(ValueError):
        chain.run(
            User=pick_best_chain.base.BasedOn("Context"),
            action=pick_best_chain.base.ToSelectFrom(actions),
            another_action=pick_best_chain.base.ToSelectFrom(actions),
        )


def test_missing_basedOn_from_throws():
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(llm=llm, prompt=PROMPT)
    actions = ["0", "1", "2"]
    with pytest.raises(ValueError):
        chain.run(action=pick_best_chain.base.ToSelectFrom(actions))


def test_ToSelectFrom_not_a_list_throws():
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(llm=llm, prompt=PROMPT)
    actions = {"actions": ["0", "1", "2"]}
    with pytest.raises(ValueError):
        chain.run(
            User=pick_best_chain.base.BasedOn("Context"),
            action=pick_best_chain.base.ToSelectFrom(actions),
        )
