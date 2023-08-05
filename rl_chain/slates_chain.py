from . import rl_chain_base as base
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from pydantic import Extra, PrivateAttr

import pandas as pd
import vowpal_wabbit_next as vw
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from itertools import chain
import random


from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from sentence_transformers import SentenceTransformer


class Label:
    chosen: List[int]
    p: List[float]
    r: Optional[float]

    def __init__(
        self, vwpred: List[List[Tuple[int, float]]], r: Optional[float] = None
    ):
        self.chosen = [p[0][0] for p in vwpred]
        self.p = [p[0][1] for p in vwpred]
        self.r = r

    def get_actions_and_probs(self):
        return zip(self.chosen, self.p)


class SlatesTextEmbedder(base.Embedder):
    """
    Slates Text Embedder class that embeds the context and actions and slates into a format that can be used by VW
    
    Attributes:
        embeddings_model (SentenceTransformer, optional): The type of embeddings to be used for feature representation. Defaults to BERT Sentence Transformer
    """

    def __init__(self, model: Optional[Any] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if model is None:
            model = SentenceTransformer("bert-base-nli-mean-tokens")

        self.model = model

    def featurize(self, raw_actions: Dict[str, List[str]]):
        # Build action embeddings
        actions = []
        actions_map = []
        for (k, v) in raw_actions.items():
            actions.append(v)
            actions_map.append(k)

        def _str(embedding):
            return " ".join([f"{i}:{e}" for i, e in enumerate(embedding)])

        action_features = [
            [
                _str(self.model.encode(action.impl))
                if isinstance(action, base._Embed)
                else action.replace(" ", "_")
                for action in slot
            ]
            for slot in actions
        ]

        return actions, actions_map, action_features

    def to_vw_format(self, inputs: Dict[str, Any]) -> str:
        slates_label = inputs.get("slates_label", None)
        named_actions = inputs.get("named_actions", None)
        if named_actions is None:
            raise ValueError("named_actions must be provided")

        actions, actions_map, action_features = self.featurize(named_actions)
        context = [f'slates shared {-1.*slates_label.r if slates_label else ""} |']
        actions = chain.from_iterable(
            [
                [f"slates action {i} |Action {action}"]
                for i, slot in enumerate(action_features)
                for action in slot
            ]
        )
        ps = (
            [f"{a}:{p}" for a, p in slates_label.get_actions_and_probs()]
            if slates_label
            else [""] * len(action_features)
        )
        slots = [f"slates slot {p} |" for p in ps]
        return "\n".join(list(chain.from_iterable([context, actions, slots])))

    def to_action_features(self, inputs: Dict[str, Any]):
        named_actions = inputs.get("named_actions", None)
        if named_actions is None:
            raise ValueError("named_actions must be provided")

        _, _, action_features = self.featurize(named_actions)
        return action_features


class Policy(ABC):
    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> Label:
        ...


class VwPolicy(Policy):
    def __init__(
        self, workspace: vw.Workspace, text_embedder: SlatesTextEmbedder, *_, **__
    ):
        self.workspace = workspace
        self.text_embedder = text_embedder

    def predict(self, inputs: Dict[str, Any]) -> Label:
        text_parser = vw.TextFormatParser(self.workspace)
        return Label(
            self.workspace.predict_one(
                base.parse_lines(text_parser, self.text_embedder.to_vw_format(inputs))
            )
        )


class RandomPolicy(Policy):
    def __init__(self, text_embedder: SlatesTextEmbedder, *_, **__):
        self.text_embedder = text_embedder

    def predict(self, inputs: Dict[str, Any]) -> Label:
        return Label(
            [
                [(random.randint(0, len(slot) - 1), 1.0 / len(slot))]
                for slot in self.text_embedder.to_action_features(inputs)
            ]
        )


class FirstChoicePolicy(Policy):
    def __init__(self, text_embedder: SlatesTextEmbedder, *_, **__):
        self.text_embedder = text_embedder

    def predict(self, inputs: Dict[str, Any]) -> Label:
        return Label(
            [[(0, 1)] for slot in self.text_embedder.to_action_features(inputs)]
        )


class LLMResponseValidatorForSlates(base.ResponseValidator):
    llm_chain: LLMChain
    prompt: PromptTemplate
    default_system_prompt = SystemMessagePromptTemplate.from_template(
        "PLEASE RESPOND ONLY WITH A SIGNLE FLOAT AND NO OTHER TEXT EXPLANATION\n You are a VERY VERY strict judge that is called on to rank a response based on given criteria.\
        You must respond with your ranking by providing a single float within the range [-1, 1], -1 being very bad response and 1 being very good response."
    )

    def __init__(self, llm, prompt=None):
        if prompt:
            self.prompt = prompt
        else:
            human_template = "Given this context {context} as the most important attribute, rank how good or bad this text selection is: {action}."
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                human_template
            )

            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    LLMResponseValidatorForSlates.default_system_prompt,
                    human_message_prompt,
                ]
            )
            self.prompt = chat_prompt

        self.llm_chain = LLMChain(llm=llm, prompt=self.prompt)

    def grade_response(
        self, inputs: Dict[str, Any], llm_response: str, **kwargs
    ) -> float:

        vars = {k: v for k, v in inputs.items() if k in self.prompt.input_variables}
        if "llm_response" in self.prompt.input_variables:
            vars["llm_response"] = llm_response
        ranking = self.llm_chain.predict(**vars)
        ranking = ranking.strip()
        try:
            resp = float(ranking)
            return resp
        except Exception:
            raise RuntimeError(
                "The llm did not manage to rank the response as expected, there is always the option to try again"
            )


class SlatesPersonalizerChain(base.RLChain):
    text_embedder: Optional[SlatesTextEmbedder] = None
    policy: Optional[Policy] = None
    _reward: List[float] = PrivateAttr(default=[])

    def __init__(self, policy=VwPolicy, *args, **kwargs):
        vw_cmd = kwargs.get("vw_cmd", [])
        if not vw_cmd:
            vw_cmd = [
                "--slates",
                "--quiet",
                "--interactions=AC",
                "--coin",
                "--squarecb",
            ]
        else:
            if "--slates" not in vw_cmd:
                raise ValueError("If vw_cmd is specified, it must include --slates")

        kwargs["vw_cmd"] = vw_cmd

        super().__init__(*args, **kwargs)
        self.text_embedder = (
            SlatesTextEmbedder() if self.text_embedder is None else self.text_embedder
        )
        self.policy = policy(self.workspace, self.text_embedder)

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return []

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        named_actions = {
            k: inputs[k] if isinstance(inputs[k], list) else [inputs[k]]
            for k in self.llm_chain.prompt.input_variables
        }

        inputs["named_actions"] = named_actions
        label = self.policy.predict(inputs=inputs)

        preds = {}
        for i, (j, a) in enumerate(zip(label.chosen, named_actions.values())):
            preds[list(named_actions.keys())[i]] = str(a[j])

        llm_resp = super()._call(run_manager=run_manager, inputs=preds)

        if self.response_validator:
            try:
                label.r = self.response_validator.grade_response(
                    inputs=preds, llm_response=llm_resp[self.output_key]
                )
                self._reward.append(label.r)

                inputs["slates_label"] = label if label else None
                vw_ex = self.text_embedder.to_vw_format(inputs)
                self._learn(vw_ex)

            except Exception as e:
                print(f"this is the error: {e}")
                base.logger.info(
                    "The LLM was not able to rank and the chain was not able to adjust to this response"
                )

        return llm_resp

    @property
    def reward(self):
        return pd.DataFrame({"r": self._reward})

    @property
    def _chain_type(self) -> str:
        return "llm_personalizer_chain"

    @classmethod
    def from_chain(cls, llm_chain: Chain, prompt: PromptTemplate, **kwargs: Any):
        return SlatesPersonalizerChain(llm_chain=llm_chain, prompt=prompt, **kwargs)

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt: PromptTemplate, **kwargs: Any):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return SlatesPersonalizerChain.from_chain(
            llm_chain=llm_chain, prompt=prompt, **kwargs
        )
