"""Chain that interprets a prompt and executes bash code to perform bash operations."""
from __future__ import annotations

import logging
import glob
import re
import os
from typing import Any, Dict, List, Optional, Tuple

import vowpal_wabbit_next as vw
from personalizer_prompt import PROMPT
from response_checker import ResponseChecker, LLMResponseCheckerForCB
from vw_example_builder import ContextualBanditTextEmbedder, Embedder
from langchain.prompts.prompt import PromptTemplate

from pydantic import Extra, PrivateAttr
import numpy as np

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from itertools import chain
from enum import Enum
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def parse_lines(parser: vw.TextFormatParser, input_str: str) -> List[vw.Example]:
    return [parser.parse_line(line) for line in input_str.split("\n")]


class PersonalizerChain(Chain):
    """
    PersonalizerChain class that utilizes the Vowpal Wabbit (VW) model for personalization.

    Attributes:
        vw_workspace_type (Type): The type of personalization algorithm to be used by the VW model.
        model_loading (bool, optional): If set to True, the chain will attempt to load an existing VW model from the latest checkpoint file in the {model_save_dir} directory (current directory if none specified). If set to False, it will start training from scratch, potentially overwriting existing files. Defaults to True.
        large_action_spaces (bool, optional): If set to True and vw_cmd has not been specified in the constructor, it will enable large action spaces
        vw_cmd (List[str], optional): Advanced users can set the VW command line to whatever they want, as long as it is compatible with the Type that is specified (Type Enum)
        model_save_dir (str, optional): The directory to save the VW model to. Defaults to the current directory.
        response_checker (ResponseChecker, optional): If set, the chain will check the response using the provided response_checker and the VW model will be updated with the result. Defaults to None.

    Notes:
        The class creates a VW model instance using the provided arguments. Before the chain object is destroyed the save_progress() function can be called. If it is called, the learned VW model is saved to a file in the current directory named `model-<checkpoint>.vw`. Checkpoints start at 1 and increment monotonically.
        When making predictions, VW is first called to choose action(s) which are then passed into the prompt with the key `{actions}`. After action selection, the LLM (Language Model) is called with the prompt populated by the chosen action(s), and the response is returned.
    """

    llm_chain: LLMChain
    workspace: Optional[vw.Workspace] = None
    next_checkpoint: int = 1
    model_save_dir: str = "./"
    response_checker: Optional[ResponseChecker] = None

    context: str = "context"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    prompt: Optional[PromptTemplate]

    class Type(Enum):
        """
        Enumeration to define the type of personalization algorithm to be used in the VW model.

        Attributes:
            CONTEXTUAL_BANDITS (tuple): Indicates the use of the Contextual Bandits algorithm.
            CONDITIONAL_CONTEXTUAL_BANDITS (tuple): Indicates the use of the Conditional Contextual Bandits algorithm.
            SLATES (tuple): Indicates Slates optimization algorithm
        """

        CONTEXTUAL_BANDITS = (1,)
        CONDITIONAL_CONTEXTUAL_BANDITS = (2,)
        SLATES = (3,)

    def __init__(
        self,
        llm: BaseLanguageModel,
        vw_workspace_type: Type,
        model_loading=True,
        large_action_spaces=False,
        vw_cmd=[],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        next_checkpoint = 1
        serialized_workspace = None

        os.makedirs(self.model_save_dir, exist_ok=True)

        if model_loading:
            vwfile = None
            files = glob.glob(f"{self.model_save_dir}/*.vw")
            pattern = r"model-(\d+)\.vw"
            highest_checkpoint = 0
            for file in files:
                match = re.search(pattern, file)
                if match:
                    checkpoint = int(match.group(1))
                    if checkpoint >= highest_checkpoint:
                        highest_checkpoint = checkpoint
                        vwfile = file

            if vwfile:
                with open(vwfile, "rb") as f:
                    serialized_workspace = f.read()

            next_checkpoint = highest_checkpoint + 1

        self.next_checkpoint = next_checkpoint
        logger.info(f"next model checkpoint = {self.next_checkpoint}")

        las_cmd = []
        if large_action_spaces and not vw_cmd:
            las_cmd = ["--large_action_space"]

        if vw_workspace_type == PersonalizerChain.Type.CONTEXTUAL_BANDITS:
            if not vw_cmd:
                vw_cmd = las_cmd + [
                    "--cb_explore_adf",
                    "--quiet",
                    "--interactions=::",
                    "--coin",
                    "--squarecb",
                ]
            else:
                if "--cb_explore_adf" not in vw_cmd:
                    raise ValueError(
                        "If vw_cmd is specified, it must include --cb_explore_adf"
                    )
        elif vw_workspace_type == PersonalizerChain.Type.CONDITIONAL_CONTEXTUAL_BANDITS:
            raise ValueError("Coming soon, not currently supported")
        elif vw_workspace_type == PersonalizerChain.Type.SLATES:
            if not vw_cmd:
                vw_cmd = las_cmd + [
                    "--slates",
                    "--quiet",
                    "--interactions=AC",
                    "--coin",
                    "--squarecb",
                ]
        else:
            raise ValueError("No other vw types supported yet")

        logger.info(f"vw command: {vw_cmd}")
        # initialize things
        if serialized_workspace:
            self.workspace = vw.Workspace(vw_cmd, model_data=serialized_workspace)
        else:
            self.workspace = vw.Workspace(vw_cmd)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.context]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        preds: Any,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        t = self.llm_chain.predict(
            **preds,
            **inputs,
            callbacks=_run_manager.get_child(),
        )
        _run_manager.on_text(t, color="green", verbose=self.verbose)
        t = t.strip()

        if self.verbose:
            _run_manager.on_text("\nCode: ", verbose=self.verbose)

        output = t
        _run_manager.on_text("\nAnswer: ", verbose=self.verbose)
        _run_manager.on_text(output, color="yellow", verbose=self.verbose)
        return {self.output_key: output}

    def save_progress(self):
        """
        This function should be called whenever there is a need to save the progress of the VW (Vowpal Wabbit) model within the chain. It saves the current state of the VW model to a file.

        File Naming Convention:
          The file will be named using the pattern `model-<checkpoint>.vw`, where `<checkpoint>` is a monotonically increasing number. The numbering starts from 1, and increments by 1 for each subsequent save. If there are already saved checkpoints, the number used for `<checkpoint>` will be the next in the sequence.

        Example:
            If there are already two saved checkpoints, `model-1.vw` and `model-2.vw`, the next time this function is called, it will save the model as `model-3.vw`.

        Note:
            Be cautious when deleting or renaming checkpoint files manually, as this could cause the function to reuse checkpoint numbers.
        """
        serialized_workspace = self.workspace.serialize()
        logger.info(
            f"storing in: {self.model_save_dir}/model-{self.next_checkpoint}.vw"
        )
        with open(f"{self.model_save_dir}/model-{self.next_checkpoint}.vw", "wb") as f:
            f.write(serialized_workspace)

    @property
    def _chain_type(self) -> str:
        return "llm_personalizer_chain"

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, prompt: PromptTemplate = PROMPT, **kwargs: Any
    ) -> ContextualBanditPersonalizerChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        if "vw_workspace_type" not in kwargs:
            raise ValueError("vw_workspace_type must be specified")
        if kwargs.get("vw_workspace_type") is PersonalizerChain.Type.CONTEXTUAL_BANDITS:
            return ContextualBanditPersonalizerChain(
                llm_chain=llm_chain, llm=llm, **kwargs
            )
        elif kwargs.get("vw_workspace_type") is PersonalizerChain.Type.SLATES:
            return SlatesPersonalizerChain(
                llm_chain=llm_chain, llm=llm, **kwargs
            )
        else:
            raise ValueError("Type not supported")


class ContextualBanditPersonalizerChain(PersonalizerChain):
    """
    ContextualBanditPersonalizerChain class that utilizes the Vowpal Wabbit (VW) model for personalization.

    The Chain is initialized with a set of potential actions. For each call to the Chain, a specific action will be chosen based on an input context.
    This chosen action is then passed to the prompt that will be utilized in the subsequent call to the LLM (Language Model).

    The flow of this chain is:
    - Chain is initialized
    - Chain is called input containing the context and the List of potential actions
    - Chain chooses an action based on the context
    - Chain calls the LLM with the chosen action
    - LLM returns a response
    - If the response_checker is specified, the response is checked against the response_checker
    - The internal model will be updated with the context, action, and reward of the response (how good or bad the response was)
    - The response is returned

    input dictionary expects:
        - context: (str, required) The context to use for personalization
        - actions: (List, required) The list of actions for the Vowpal Wabbit model to choose from. This list can either be a List of str's or a List of Dict's.
                - Actions provided as a list of strings e.g. actions = ["action1", "action2", "action3"]
                - If actions are provided as a list of dictionaries, each action should be a dictionary where the keys are namespace names and the values are the corresponding action strings e.g. actions = [{"namespace1": "action1", "namespace2": "action2"}, {"namespace1": "action3", "namespace2": "action4"}]    
    Extends:
        PersonalizerChain

    Attributes:
        text_embedder: (ContextualBanditTextEmbedder, optional) The text embedder to use for embedding the context and the actions. If not provided, a default embedder is used.
    """

    class ResponseResult:
        def __init__(
            self,
            chosen_action: int,
            chosen_action_probability: float,
            cost: Optional[float],
            inputs: Dict[str, Any],
        ):
            self.chosen_action = chosen_action
            self.chosen_action_probability = chosen_action_probability
            self.cost = cost
            self.inputs = inputs

    latest_response: Optional[ResponseResult] = None
    text_embedder: ContextualBanditTextEmbedder = ContextualBanditTextEmbedder(
        "bert-base-nli-mean-tokens"
    )
    actions: str = "actions"  #: :meta private:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.actions, self.context]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        When chain.run() is called with the given inputs, this function is called. It is responsible for calling the VW model to choose an action based on the `context`, and then calling the LLM (Language Model) with the chosen action to generate a response.

        Attributes:
            inputs: (Dict, required) The inputs to the chain. The inputs must contain a keys `context` and `actions`. That is the context to be used for selecting an action that will be passed to the LLM prompt.
            run_manager: (CallbackManagerForChainRun, optional) The callback manager to use for this run. If not provided, a default callback manager is used.
            
        Returns:
            A dictionary containing:
                - `response`: The response generated by the LLM (Language Model).
                - `response_result`: A ResponseResult object containing all the information needed to learn the reward for the chosen action at a later point. If an automatic response_checker is not provided, then this object can be used at a later point with the `learn_delayed_reward()` function to learn the delayed reward and update the Vowpal Wabbit model.
                    - the `cost` in the `response_result` object is set to None if an automatic response_checker is not provided or if the response_checker failed (e.g. LLM timeout or LLM failed to rank correctly).
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        if self.workspace is None:
            raise RuntimeError("Workspace must be set before calling the chain")

        text_parser = vw.TextFormatParser(self.workspace)

        actions = inputs[self.actions]
        vw_ex = self.text_embedder.to_vw_format(inputs=inputs)
        multi_ex = parse_lines(text_parser, vw_ex)
        preds: List[Tuple[int, float]] = self.workspace.predict_one(multi_ex)
        prob_sum = sum(prob for _, prob in preds)
        probabilities = [prob / prob_sum for _, prob in preds]

        ## explore
        sampled_index = np.random.choice(len(preds), p=probabilities)
        sampled_ap = preds[sampled_index]
        sampled_action = sampled_ap[0]
        sampled_prob = sampled_ap[1]

        pred_action = actions[sampled_action]

        llm_resp: Dict[str, Any] = super()._call(
            run_manager=run_manager, inputs=inputs, preds={'chosen_actions': [pred_action]}
        )

        # llm_resp: Dict[str, Any] = {self.output_key : ""}

        latest_cost = None

        if self.response_checker:
            try:
                cost = -1.0 * self.response_checker.grade_response(
                    inputs=inputs,
                    llm_response=llm_resp[self.output_key],
                    chosen_action=pred_action,
                )
                latest_cost = cost
                text_parser = vw.TextFormatParser(self.workspace)
                cb_label = (sampled_action, cost, sampled_prob)

                vw_ex = self.text_embedder.to_vw_format(
                    cb_label=cb_label,
                    inputs=inputs,
                )
                multi_ex = parse_lines(text_parser, vw_ex)
                self.workspace.learn_one(multi_ex)

            except Exception as e:
                logger.info(
                    f"The LLM was not able to rank and the chain was not able to adjust to this response. Error: {e}"
                )

        self.latest_response = ContextualBanditPersonalizerChain.ResponseResult(
            chosen_action=sampled_action,
            chosen_action_probability=sampled_prob,
            inputs=inputs,
            cost=latest_cost,
        )

        llm_resp[self.output_key] = {
            "response": llm_resp[self.output_key],
            "response_result": self.latest_response,
        }

        return llm_resp

    def learn_delayed_reward(
        self, reward: float, response_result: ResponseResult, force_reward=False
    ):
        """
        Learn will be called with the reward specified and the actions/embeddings/etc stored in response_result

        Will raise an error if check_response is set to True and force_reward=True was not provided during the method call
        force_cost should be used if the check_response failed to check the response correctly
        """
        if self.response_checker and not force_reward:
            raise RuntimeError(
                "check_response is set to True, this must be turned off for explicit feedback and training to be provided, or overriden by calling the method with force_reward=True"
            )
        text_parser = vw.TextFormatParser(self.workspace)
        cost = -1.0 * reward

        cb_label = (
            response_result.chosen_action,
            cost,
            response_result.chosen_action_probability,
        )

        vw_ex = self.text_embedder.to_vw_format(
            cb_label=cb_label,
            inputs=response_result.inputs,
        )

        multi_ex = parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)


class _Embed:
    def __init__(self, impl):
        self.impl = impl

    def __str__(self):
        return self.impl

def Embed(anything):
    if isinstance(anything, list):
        return [_Embed(v) for v in anything]
    return _Embed(anything)

class SlatesPersonalizerChain(PersonalizerChain):
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
        label: Optional[SlatesPersonalizerChain.Label]

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

    last_decision: Optional[Decision] = None

    action_embeddings: List[List[str]] = []
    actions: List[List[str]] = []
    actions_map: List[str] = []
    embeddings_model: Optional[SentenceTransformer] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings_model = SentenceTransformer("bert-base-nli-mean-tokens")

    def set_actions(self, actions: Dict[str, List[str]]):
        """
        At any time new actions can be set by this function call

        Attributes:
            actions: a list of list strings containing the actions that will be transformed to embeddings using the FeatureEmbeddings
        """
        # Build action embeddings
        for (i, (k, v)) in enumerate((actions.items())):
            self.actions.append(v)
            self.actions_map.append(k)
            
        def _str(embedding):
            return ' '.join([f'{i}:{e}' for i, e in enumerate(embedding)])
        
        self.action_embeddings = [
            [_str(self.embeddings_model.encode(action.impl)) if isinstance(action, _Embed) else action for action in slot] for slot in self.actions]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        self.set_actions({k: inputs[k] if isinstance(inputs[k], list) else [inputs[k]] for k in self.llm_chain.prompt.input_variables})
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        text_parser = vw.TextFormatParser(self.workspace)
        context = inputs[self.context]

        self.last_decision = self.Decision(
            self.action_embeddings)

        self.last_decision.label = self.Label(
            self.workspace.predict_one(parse_lines(text_parser, self.last_decision.vwtxt)))
        preds = {}
        for i, (j, actions) in enumerate(zip(self.last_decision.label.chosen, self.actions)):
            preds[self.actions_map[i]] = str(actions[j]) 
        llm_resp = super()._call(
            run_manager=run_manager, inputs=preds, preds={}
        )

        if self.response_checker:
            try:
                self.last_decision.label.r = self.response_checker.grade_response(
                    inputs=preds, llm_response=llm_resp[self.output_key]
                )
                self.workspace.learn_one(parse_lines(text_parser, self.last_decision.vwtxt))

            except Exception as e:
                print(f"this is the error: {e}")
                logger.info(
                    "The LLM was not able to rank and the chain was not able to adjust to this response"
                )

        return llm_resp

    def learn_with_specific_cost(self, cost: int, force_cost=False):
        """
        Learn will be called with the cost specified
        Will raise an error if check_response is set to True and force_cost=True was not provided during the method call
        force_cost should be used if the check_response failed to check the response correctly
        """
        if self.check_response and not force_cost:
            raise RuntimeError(
                "check_response is set to True, this must be turned off for explicit feedback and training to be provided, or overriden by calling the method with force_cost=True"
            )
        self.last_decision.label.r = -cost
        text_parser = vw.TextFormatParser(self.workspace)
        self.workspace.learn_one(parse_lines(text_parser, self.last_decision.vwtxt))


# ### TODO:
# - persist data to log file?
# - would this work with a longer chain?
# - fix save_progress to not override existing file
# - Naming: is LLMResponseChecker a good enough name?, Personalizer? CB how should they be named for a good API?
# - Good documentation: check langchain requirements we are adding explanations on the functions as we go
# - be able to specify vw model file name