import sys

sys.path.append("..")

import rl_chain.pick_best_chain as pick_best_chain
from test_utils import MockEncoder
import pytest
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import FakeListChatModel

encoded_text = "[ e n c o d e d ] "


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


def test_default_embeddings():
    llm, PROMPT = setup()
    text_embedder = pick_best_chain.PickBestTextEmbedder(model=MockEncoder())
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, text_embedder=text_embedder
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = encoded_text + " ".join(char for char in str1)
    encoded_str2 = encoded_text + " ".join(char for char in str2)
    encoded_str3 = encoded_text + " ".join(char for char in str3)

    ctx_str_1 = "context1"
    ctx_str_2 = "context2"

    encoded_ctx_str_1 = encoded_text + " ".join(char for char in ctx_str_1)
    encoded_ctx_str_2 = encoded_text + " ".join(char for char in ctx_str_2)

    expected = f"""shared |User {ctx_str_1 + " " + encoded_ctx_str_1} \n|action {str1 + " " + encoded_str1} \n|action {str2 + " " + encoded_str2} \n|action {str3 + " " + encoded_str3} """

    actions = [str1, str2, str3]

    response = chain.run(
        User=pick_best_chain.base.BasedOn(ctx_str_1),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    decision_metadata = response["decision_metadata"]
    vw_str = text_embedder.to_vw_format(decision_metadata)
    assert vw_str == expected


def test_default_embeddings_off():
    llm, PROMPT = setup()
    text_embedder = pick_best_chain.PickBestTextEmbedder(model=MockEncoder())
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, text_embedder=text_embedder, auto_embed=False
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    ctx_str_1 = "context1"

    expected = f"""shared |User {ctx_str_1} \n|action {str1} \n|action {str2} \n|action {str3} """

    actions = [str1, str2, str3]

    response = chain.run(
        User=pick_best_chain.base.BasedOn(ctx_str_1),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    decision_metadata = response["decision_metadata"]
    vw_str = text_embedder.to_vw_format(decision_metadata)
    assert vw_str == expected


def test_default_embeddings_mixed_w_explicit_user_embeddings():
    llm, PROMPT = setup()
    text_embedder = pick_best_chain.PickBestTextEmbedder(model=MockEncoder())
    chain = pick_best_chain.PickBest.from_llm(
        llm=llm, prompt=PROMPT, text_embedder=text_embedder
    )

    str1 = "0"
    str2 = "1"
    str3 = "2"
    encoded_str1 = encoded_text + " ".join(char for char in str1)
    encoded_str2 = encoded_text + " ".join(char for char in str2)
    encoded_str3 = encoded_text + " ".join(char for char in str3)

    ctx_str_1 = "context1"
    ctx_str_2 = "context2"

    encoded_ctx_str_1 = encoded_text + " ".join(char for char in ctx_str_1)
    encoded_ctx_str_2 = encoded_text + " ".join(char for char in ctx_str_2)

    expected = f"""shared |User {encoded_ctx_str_1} |User2 {ctx_str_2 + " " + encoded_ctx_str_2} \n|action {str1 + " " + encoded_str1} \n|action {str2 + " " + encoded_str2} \n|action {encoded_str3} """

    actions = [str1, str2, pick_best_chain.base.Embed(str3)]

    response = chain.run(
        User=pick_best_chain.base.BasedOn(pick_best_chain.base.Embed(ctx_str_1)),
        User2=pick_best_chain.base.BasedOn(ctx_str_2),
        action=pick_best_chain.base.ToSelectFrom(actions),
    )
    decision_metadata = response["decision_metadata"]
    vw_str = text_embedder.to_vw_format(decision_metadata)
    assert vw_str == expected
