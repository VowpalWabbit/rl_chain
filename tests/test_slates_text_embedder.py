import sys

sys.path.append("..")

import rl_chain.slates_chain as slates
from test_utils import MockEncoder

import pytest

encoded_keyword = "[encoded]"


def test_slate_text_creation_no_label_no_emb():
    named_actions = {"prefix": ["0", "1"], "context": ["bla"], "suffix": ["0", "1"]}
    expected = """slates shared  |\nslates action 0 |Action 0\nslates action 0 |Action 1\nslates action 1 |Action bla\nslates action 2 |Action 0\nslates action 2 |Action 1\nslates slot  |\nslates slot  |\nslates slot  |"""
    feature_embedder = slates.SlatesFeatureEmbedder()
    event = slates.SlatesPersonalizerChain.Event(
        inputs={}, to_select_from=named_actions, based_on={}
    )
    vw_str_ex = feature_embedder.format(event)
    assert vw_str_ex == expected


def _str(embedding):
    return " ".join([f"{i}:{e}" for i, e in enumerate(embedding)])


def test_slate_text_creation_no_label_w_emb():
    action00 = "0"
    action01 = "1"
    action10 = "bla"
    action20 = "0"
    action21 = "1"
    encoded_action00 = _str(encoded_keyword + action00)
    encoded_action01 = _str(encoded_keyword + action01)
    encoded_action10 = _str(encoded_keyword + action10)
    encoded_action20 = _str(encoded_keyword + action20)
    encoded_action21 = _str(encoded_keyword + action21)

    named_actions = {
        "prefix": slates.base.Embed(["0", "1"]),
        "context": slates.base.Embed(["bla"]),
        "suffix": slates.base.Embed(["0", "1"]),
    }
    expected = f"""slates shared  |\nslates action 0 |Action {encoded_action00}\nslates action 0 |Action {encoded_action01}\nslates action 1 |Action {encoded_action10}\nslates action 2 |Action {encoded_action20}\nslates action 2 |Action {encoded_action21}\nslates slot  |\nslates slot  |\nslates slot  |"""
    feature_embedder = slates.SlatesFeatureEmbedder(model=MockEncoder())
    event = slates.SlatesPersonalizerChain.Event(
        inputs={}, to_select_from=named_actions, based_on={}
    )
    vw_str_ex = feature_embedder.format(event)
    assert vw_str_ex == expected
