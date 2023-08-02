import sys
sys.path.append("..")

import rl_chain.rl_chain_base as base
from test_utils import MockEncoder

encoded_text = "[ e n c o d e d ] "

def test_simple_context_str_no_emb():
    expected = [{"default_namespace": "test"}]
    assert(base.embed("test", MockEncoder(), "default_namespace") == expected)


def test_simple_context_str_w_emb():
    str1 = "test"
    encoded_str1 = " ".join(char for char in str1)
    expected = [{"default_namespace": encoded_text + encoded_str1}]
    assert(base.embed(base.Embed(str1), MockEncoder(), "default_namespace") == expected)

def test_context_w_namespace_no_emb():
    expected = [{"test_namespace": "test"}]
    assert(base.embed({"test_namespace": "test"}, MockEncoder()) == expected)

def test_context_w_namespace_w_emb():
    str1 = "test"
    encoded_str1 = " ".join(char for char in str1)
    expected = [{"test_namespace": encoded_text + encoded_str1}]
    assert(base.embed({"test_namespace": base.Embed(str1)}, MockEncoder()) == expected)

def test_context_w_namespace_w_emb2():
    str1 = "test"
    encoded_str1 = " ".join(char for char in str1)
    expected = [{"test_namespace": encoded_text + encoded_str1}]
    assert(base.embed(base.Embed({"test_namespace": str1}), MockEncoder()) == expected)

def test_context_w_namespace_w_some_emb():
    str1 = "test1"
    str2 = "test2"
    encoded_str2 = " ".join(char for char in str2)
    expected = [{"test_namespace": str1}, {"test_namespace2": encoded_text + encoded_str2}]
    assert(base.embed({"test_namespace": str1, "test_namespace2": base.Embed(str2)}, MockEncoder()) == expected)

def test_simple_action_strlist_no_emb():
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    expected = [{"default_namespace": str1}, {"default_namespace": str2}, {"default_namespace": str3}]
    assert(base.embed([str1, str2, str3], MockEncoder(), "default_namespace") == expected)

def test_simple_action_strlist_w_emb():
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str1 = " ".join(char for char in str1)
    encoded_str2 = " ".join(char for char in str2)
    encoded_str3 = " ".join(char for char in str3)
    expected = [{"default_namespace": encoded_text + encoded_str1}, {"default_namespace": encoded_text + encoded_str2}, {"default_namespace": encoded_text + encoded_str3}]
    assert(base.embed(base.Embed([str1, str2, str3]), MockEncoder(), "default_namespace") == expected)

def test_simple_action_strlist_w_some_emb():
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str2 = " ".join(char for char in str2)
    encoded_str3 = " ".join(char for char in str3)
    expected = [{"default_namespace": str1}, {"default_namespace": encoded_text + encoded_str2}, {"default_namespace": encoded_text + encoded_str3}]
    assert(base.embed([str1, base.Embed(str2), base.Embed(str3)], MockEncoder(), "default_namespace") == expected)


def test_action_strlist_w_namespace_no_emb():
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    expected = [{"test_namespace": str1}, {"test_namespace": str2}, {"test_namespace": str3}]
    assert(base.embed([{"test_namespace": str1}, {"test_namespace": str2}, {"test_namespace": str3}], MockEncoder()) == expected)

def test_action_strlist_w_namespace_w_emb():
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str1 = " ".join(char for char in str1)
    encoded_str2 = " ".join(char for char in str2)
    encoded_str3 = " ".join(char for char in str3)
    expected = [{"test_namespace": encoded_text + encoded_str1}, {"test_namespace": encoded_text + encoded_str2}, {"test_namespace": encoded_text + encoded_str3}]
    assert(base.embed([{"test_namespace": base.Embed(str1)}, {"test_namespace": base.Embed(str2)}, {"test_namespace": base.Embed(str3)}], MockEncoder()) == expected)

def test_action_strlist_w_namespace_w_emb2():
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str1 = " ".join(char for char in str1)
    encoded_str2 = " ".join(char for char in str2)
    encoded_str3 = " ".join(char for char in str3)
    expected = [{"test_namespace1": encoded_text + encoded_str1}, {"test_namespace2": encoded_text + encoded_str2}, {"test_namespace3": encoded_text + encoded_str3}]
    assert(base.embed(base.Embed([{"test_namespace1": str1}, {"test_namespace2": str2}, {"test_namespace3": str3}]), MockEncoder()) == expected)

def test_action_strlist_w_namespace_w_some_emb():
    str1 = "test1"
    str2 = "test2"
    str3 = "test3"
    encoded_str2 = " ".join(char for char in str2)
    encoded_str3 = " ".join(char for char in str3)
    expected = [{"test_namespace": str1}, {"test_namespace": encoded_text + encoded_str2}, {"test_namespace": encoded_text + encoded_str3}]
    assert(base.embed([{"test_namespace": str1}, {"test_namespace": base.Embed(str2)}, {"test_namespace": base.Embed(str3)}], MockEncoder()) == expected)