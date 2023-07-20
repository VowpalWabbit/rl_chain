from .rl_chain_base import *
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import pandas as pd
import vowpal_wabbit_next as vw
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
from itertools import chain
import random


from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from sentence_transformers import SentenceTransformer

#TODO: remove copy-paste
def parse_lines(parser: vw.TextFormatParser, input_str: str) -> List[vw.Example]:
    return [parser.parse_line(line) for line in input_str.split("\n")]


class Label:
    chosen: List[int]
    p: List[float]
    r: Optional[float]

    def __init__(self, vwpred: List[List[Tuple[int, float]]], r: Optional[float] = None):
        self.chosen = [p[0][0] for p in vwpred]
        self.p = [p[0][1] for p in vwpred]
        self.r = r

    def ap(self):
        return zip(self.chosen, self.p)


class Decision:
    actions: List[List[str]]
    label: Optional[Label]

    def __init__(self, actions, label=None):
        self.actions = actions
        self.label = label

    @property
    def vwtxt(self):
        context = [f'slates shared {-self.label.r if self.label else ""} |']
        actions = chain.from_iterable([[
            f'slates action {i} |Action {action}'] 
            for i, slot in enumerate(self.actions) for action in slot])
        ps = [f'{a}:{p}' for a, p in self.label.ap()] if self.label else [''] * len(self.actions)
        slots = [f'slates slot {p} |' for p in ps]
        return '\n'.join(list(chain.from_iterable([context, actions, slots]))) # TODO: remove


class Policy(ABC):
    @abstractmethod
    def predict(self, decision: Decision) -> Label:
        ...


class VwPolicy(Policy):
    def __init__(self, workspace: vw.Workspace, *_, **__):
        self.workspace = workspace
    
    def predict(self, decision: Decision) -> Label:
        text_parser = vw.TextFormatParser(self.workspace)
        return Label(self.workspace.predict_one(parse_lines(text_parser, decision.vwtxt)))


class RandomPolicy(Policy):
    def __init__(self, *_, **__):
        ...

    def predict(self, decision: Decision) -> Label:
        return Label([[(random.randint(0, len(slot) - 1), 1.0 / len(slot))] for slot in decision.actions])


class FirstChoicePolicy(Policy):
    def __init__(self, *_, **__):
        ...

    def predict(self, decision: Decision) -> Label:
        return Label([[(0, 1)] for slot in decision.actions])

class _Embed:
    def __init__(self, impl):
        self.impl = impl

    def __str__(self):
        return self.impl

def Embed(anything):
    if isinstance(anything, list):
        return [_Embed(v) for v in anything]
    return _Embed(anything)

class LLMResponseValidatorForSlates(ResponseValidator):
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
                [LLMResponseValidatorForSlates.default_system_prompt, human_message_prompt]
            )
            self.prompt = chat_prompt

        self.llm_chain = LLMChain(llm=llm, prompt=self.prompt)

    def grade_response(
        self, inputs: Dict[str, Any], llm_response: str, **kwargs
    ) -> float:
        
        vars = {k: v for k, v in inputs.items() if k in self.prompt.input_variables}
        if 'llm_response' in self.prompt.input_variables:
            vars['llm_response'] = llm_response
        ranking = self.llm_chain.predict(**vars)
        ranking = ranking.strip()
        try:
            resp = float(ranking)
            return resp
        except Exception:
            raise RuntimeError(
                "The llm did not manage to rank the response as expected, there is always the option to try again"
            )

class SlatesPersonalizerChain(RLChain):
    last_decision: Optional[Decision] = None
    embeddings_model: Optional[SentenceTransformer] = None
    policy: Optional[Policy] = None
    _reward: List[float] = PrivateAttr(default=[])

    def __init__(self, policy = VwPolicy, *args, **kwargs):
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
                raise ValueError(
                    "If vw_cmd is specified, it must include --slates"
                )

        kwargs["vw_cmd"] = vw_cmd

        super().__init__(*args, **kwargs)
        self.embeddings_model = SentenceTransformer("bert-base-nli-mean-tokens")
        self.policy = policy(self.workspace)

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return []

    def _featurize(self, raw_actions: Dict[str, List[str]]):
        """
        At any time new actions can be set by this function call

        Attributes:
            actions: a list of list strings containing the actions that will be transformed to embeddings using the FeatureEmbeddings
        """
        # Build action embeddings
        actions = []
        actions_map = []
        for (k, v) in raw_actions.items():
            actions.append(v)
            actions_map.append(k)
            
        def _str(embedding):
            return ' '.join([f'{i}:{e}' for i, e in enumerate(embedding)])
        
        action_features = [
            [_str(self.embeddings_model.encode(action.impl)) if isinstance(action, _Embed) else action.replace(" ", "_") for action in slot] for slot in actions]
        return actions, actions_map, action_features

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        named_actions = {k: inputs[k] if isinstance(inputs[k], list) else [inputs[k]] for k in self.llm_chain.prompt.input_variables}
        actions, actions_map, action_features = self._featurize(named_actions)        
        self.last_decision = Decision(action_features)
        self.last_decision.label = self.policy.predict(self.last_decision)

        preds = {}
        for i, (j, a) in enumerate(zip(self.last_decision.label.chosen, actions)):
            preds[actions_map[i]] = str(a[j]) 
        llm_resp = super()._call(run_manager=run_manager, inputs=preds)

        if self.response_validator:
            try:
                self.last_decision.label.r = self.response_validator.grade_response(
                    inputs=preds, llm_response=llm_resp[self.output_key]
                )
                self._reward.append(self.last_decision.label.r)
                self._learn(self.last_decision.vwtxt)

            except Exception as e:
                print(f"this is the error: {e}")
                logger.info(
                    "The LLM was not able to rank and the chain was not able to adjust to this response"
                )

        return llm_resp

    def learn_with_specific_cost(self, cost: int, force_cost=False):
        ... # TODO: implement

    @property
    def reward(self):
        return pd.DataFrame({'r': self._reward})

    @property
    def _chain_type(self) -> str:
        return "llm_personalizer_chain"

    @classmethod
    def from_chain(cls, llm_chain: Chain, prompt: PromptTemplate, **kwargs: Any):
        return SlatesPersonalizerChain(
            llm_chain=llm_chain, prompt=prompt, **kwargs
        )

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt: PromptTemplate, **kwargs: Any):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return SlatesPersonalizerChain.from_chain(llm_chain=llm_chain, prompt=prompt, **kwargs)
