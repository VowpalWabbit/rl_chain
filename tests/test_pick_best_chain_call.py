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


def test_update_with_delayed_score_with_auto_validator_throws():
    llm, PROMPT = setup()
    # this LLM returns a number so that the auto validator will return that
    auto_val_llm = FakeListChatModel(responses=["3"])
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=pick_best_chain.PickBestAutoSelectionScorer(llm=auto_val_llm),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score == 3.0
    with pytest.raises(RuntimeError):
        chain.update_with_delayed_score(event=selection_metadata, score=100)


def test_update_with_delayed_score_force():
    llm, PROMPT = setup()
    # this LLM returns a number so that the auto validator will return that
    auto_val_llm = FakeListChatModel(responses=["3"])
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm,
        prompt=PROMPT,
        selection_scorer=pick_best_chain.PickBestAutoSelectionScorer(llm=auto_val_llm),
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score == 3.0
    chain.update_with_delayed_score(
        event=selection_metadata, score=100, force_score=True
    )
    assert selection_metadata.selected.score == 100.0


def test_update_with_delayed_score():
    llm, PROMPT = setup()
    chain = pick_best_chain.PickBest.from_llm(llm=llm, prompt=PROMPT)
    actions = ["0", "1", "2"]
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score == None
    chain.update_with_delayed_score(event=selection_metadata, score=100)
    assert selection_metadata.selected.score == 100.0


def test_user_defined_scorer():
    llm, PROMPT = setup()

    class CustomSelectionScorer(pick_best_chain.base.SelectionScorer):
        def score_response(self, inputs, llm_response: str) -> float:
            score = 200
            return score

    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, selection_scorer=CustomSelectionScorer()
    )
    actions = ["0", "1", "2"]
    response = chain.run(
        User=pick_best_chain.base.BasedOn("Context"),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    assert response["response"] == "hey"
    selection_metadata = response["selection_metadata"]
    assert selection_metadata.selected.score == 200.0
