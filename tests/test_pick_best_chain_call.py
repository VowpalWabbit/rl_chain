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


def test_learn_delayed_reward_with_auto_validator_throws():
    llm, PROMPT = setup()
    # this LLM returns a number so that the auto validator will return that
    auto_val_llm = FakeListChatModel(responses=["3"])
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        response_validator=pick_best_chain.PickBestAutoResponseValidator(
            llm=auto_val_llm
        ),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    decision_metadata = response["decision_metadata"]
    assert decision_metadata.label.cost == -3.0
    with pytest.raises(RuntimeError):
        chain.learn_delayed_reward(event=decision_metadata, reward=100)


def test_learn_delayed_reward_force():
    llm, PROMPT = setup()
    # this LLM returns a number so that the auto validator will return that
    auto_val_llm = FakeListChatModel(responses=["3"])
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        response_validator=pick_best_chain.PickBestAutoResponseValidator(
            llm=auto_val_llm
        ),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    decision_metadata = response["decision_metadata"]
    assert decision_metadata.label.cost == -3.0
    chain.learn_delayed_reward(event=decision_metadata, reward=100, force_reward=True)
    assert decision_metadata.label.cost == -100.0


def test_learn_delayed_reward():
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(llm=llm, prompt=PROMPT)
    actions = ["0", "1", "2"]
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    decision_metadata = response["decision_metadata"]
    assert decision_metadata.label.cost == None
    chain.learn_delayed_reward(event=decision_metadata, reward=100)
    assert decision_metadata.label.cost == -100.0


def test_user_defined_reward():
    llm, PROMPT = setup()

    class CustomResponseValidator(pick_best_chain.base.ResponseValidator):
        def grade_response(self, inputs, llm_response: str) -> float:
            reward = 200
            return reward

    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, response_validator=CustomResponseValidator()
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    decision_metadata = response["decision_metadata"]
    assert decision_metadata.label.cost == -200.0
