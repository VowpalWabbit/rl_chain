from __future__ import annotations

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


class SlatesLabel(base.Label):
    chosen: List[int]
    p: List[float]
    cost: Optional[float]

    def __init__(
        self, vwpred: List[List[Tuple[int, float]]], cost: Optional[float] = None
    ):
        self.chosen = [p[0][0] for p in vwpred]
        self.p = [p[0][1] for p in vwpred]
        self.cost = cost

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

    def to_action_features(self, actions: Dict[str, Any]):
        def _str(embedding):
            return " ".join([f"{i}:{e}" for i, e in enumerate(embedding)])

        action_features = [
            [
                _str(self.model.encode(action.value))
                if isinstance(action, base._Embed)
                else action.replace(" ", "_")
                for action in slot
            ]
            for slot in actions.values()
        ]

        return action_features

    def to_vw_format(self, vw_event: SlatesPersonalizerChain.Event) -> str:
        action_features = self.to_action_features(vw_event.actions)

        cost = vw_event.label.cost if vw_event.label else ""
        context_str = f"slates shared {cost} "

        if vw_event.context:
            embedded_context = base.embed(vw_event.context, self.model)
            for context_item in embedded_context:
                for ns, ctx in context_item.items():
                    context_str += (
                        f"|{ns} {' '.join(ctx) if isinstance(ctx, list) else ctx} "
                    )
        else:
            context_str += "|"  # empty context

        actions = chain.from_iterable(
            [
                [f"slates action {i} |Action {action}"]
                for i, slot in enumerate(action_features)
                for action in slot
            ]
        )
        ps = (
            [f"{a}:{p}" for a, p in vw_event.label.get_actions_and_probs()]
            if vw_event.label
            else [""] * len(action_features)
        )
        slots = [f"slates slot {p} |" for p in ps]
        return "\n".join(list(chain.from_iterable([[context_str], actions, slots])))


class RandomPolicy(base.Policy):
    def __init__(self, text_embedder: base.Embedder, *_, **__):
        self.text_embedder = text_embedder

    def predict(self, event: SlatesPersonalizerChain.Event) -> Any:
        return [
            [(random.randint(0, len(slot) - 1), 1.0 / len(slot))]
            for slot in self.text_embedder.to_action_features(event.actions)
        ]

    def learn(self, event: SlatesPersonalizerChain.Event) -> Any:
        pass

    def log(self, event: SlatesPersonalizerChain.Event) -> Any:
        pass


class FirstChoicePolicy(base.Policy):
    def __init__(self, text_embedder: base.Embedder, *_, **__):
        self.text_embedder = text_embedder

    def predict(self, event: SlatesPersonalizerChain.Event) -> Any:
        return [
            [(0, 1)] for slot in self.text_embedder.to_action_features(event.actions)
        ]

    def learn(self, event: SlatesPersonalizerChain.Event) -> Any:
        pass

    def log(self, event: SlatesPersonalizerChain.Event) -> Any:
        pass


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

    def grade_response(self, inputs: Dict[str, Any], llm_response: str) -> float:
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
    class Event(base.Event):
        def __init__(
            self,
            inputs: Dict[str, Any],
            actions: Dict[str, Any],
            context: Dict[str, Any],
            label: Optional[SlatesLabel] = None,
        ):
            super().__init__(inputs=inputs, label=label)
            self.actions = actions
            self.context = context

    _reward: List[float] = PrivateAttr(default=[])

    def __init__(self, text_embedder: Optional[base.Embedder] = None, *args, **kwargs):
        vw_cmd = kwargs.get("vw_cmd", [])
        if not vw_cmd:
            vw_cmd = [
                "--slates",
                "--quiet",
                "--interactions=::",
                "--coin",
                "--squarecb",
            ]
        else:
            if "--slates" not in vw_cmd:
                raise ValueError("If vw_cmd is specified, it must include --slates")

        kwargs["vw_cmd"] = vw_cmd

        if text_embedder is None:
            text_embedder = SlatesTextEmbedder()

        super().__init__(text_embedder=text_embedder, *args, **kwargs)

    def _get_context_and_actions(self, inputs: Dict[str, Any]):
        named_actions = {
            k: inputs[k].value
            for k in inputs.keys()
            if isinstance(inputs[k], base._ToSelectFrom)
        }

        if not named_actions:
            raise ValueError(
                "No variables using 'ToSelectFrom' found in the inputs. Please include at least one variable containing a list to select from."
            )

        context = {
            k: inputs[k].value
            if isinstance(inputs[k].value, list)
            else [inputs[k].value]
            for k in inputs.keys()
            if isinstance(inputs[k], base._BasedOn)
        }

        return context, named_actions

    def call_before_predict(
        self, inputs: Dict[str, Any]
    ) -> SlatesPersonalizerChain.Event:
        context, named_actions = self._get_context_and_actions(inputs)
        event = SlatesPersonalizerChain.Event(
            inputs=inputs, actions=named_actions, context=context
        )
        return event

    def call_after_predict_before_llm(
        self,
        inputs: Dict[str, Any],
        event: SlatesPersonalizerChain.Event,
        vwpreds: List[List[Tuple[int, float]]],
    ) -> Tuple[Dict[str, Any], SlatesPersonalizerChain.Event]:
        label = SlatesLabel(vwpred=vwpreds)
        event.label = label

        preds = {}
        for i, (j, a) in enumerate(zip(label.chosen, event.actions.values())):
            preds[list(event.actions.keys())[i]] = str(a[j])

        next_chain_inputs = inputs.copy()
        next_chain_inputs.update(preds)

        return next_chain_inputs, event

    def call_after_llm_before_scoring(
        self, llm_response: str, event: SlatesPersonalizerChain.Event
    ) -> Tuple[Dict[str, Any], SlatesPersonalizerChain.Event]:
        return event.inputs, event

    def call_after_scoring_before_learning(
        self, llm_response: str, event: Event, response_quality: Optional[float]
    ) -> SlatesPersonalizerChain.Event:
        event.label.cost = (
            -1.0 * response_quality if response_quality is not None else None
        )
        self._reward.append(response_quality)
        return event

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        return super()._call(run_manager=run_manager, inputs=inputs)

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
