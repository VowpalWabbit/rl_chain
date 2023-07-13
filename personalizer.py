"""Chain that interprets a prompt and executes bash code to perform bash operations."""
from __future__ import annotations

import logging
import glob
import re
import os
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
import vowpal_wabbit_next as vw
from personalizer_prompt import PROMPT
from response_checker import ResponseChecker, LLMResponseChecker
from langchain.prompts.prompt import PromptTemplate

from pydantic import Extra, PrivateAttr
import numpy as np

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from enum import Enum

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
        embeddings_model (SentenceTransformer, optional): The type of embeddings to be used for feature representation. Defaults to BERT.
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
    embeddings_model: SentenceTransformer = SentenceTransformer(
        "bert-base-nli-mean-tokens"
    )
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
            actions=[preds],
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
        cls,
        llm: BaseLanguageModel,
        prompt: PromptTemplate = PROMPT,
        **kwargs: Any,
    ) -> ContextualBanditPersonalizerChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        if "vw_workspace_type" not in kwargs:
            raise ValueError("vw_workspace_type must be specified")
        if kwargs.get("vw_workspace_type") is PersonalizerChain.Type.CONTEXTUAL_BANDITS:
            return ContextualBanditPersonalizerChain(
                llm_chain=llm_chain, llm=llm, **kwargs
            )
        elif (
            kwargs.get("vw_workspace_type")
            is PersonalizerChain.Type.CONDITIONAL_CONTEXTUAL_BANDITS
        ):
            raise ValueError("Not implemented yet")
        else:
            raise ValueError("Type not supported")


class ContextualBanditPersonalizerChain(PersonalizerChain):
    class ResponseResult:
        def __init__(
            self,
            action_embeddings: List[str],
            chosen_action: int,
            chosen_action_probability: float,
            context_emb: str,
            cost: Optional[float],
        ):
            self.action_embeddings = action_embeddings
            self.chosen_action = chosen_action
            self.chosen_action_probability = chosen_action_probability
            self.context_emb = context_emb
            self.cost = cost

    latest_response: Optional[ResponseResult] = None
    action_embeddings: List[str] = []
    actions: List[str] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_actions(self.actions)

    def set_actions_and_embeddings(self, actions: List[str], action_embeddings: List):
        """
        At any time new actions and their embeddings can be set by this function call

        Attributes:
            actions: a list of strings containing the actions
            action_embeddings: a list containing the embeddings of the action strings
        """
        self.actions = actions
        self.action_embeddings = action_embeddings

    def set_actions(self, actions: List[str]):
        """
        At any time new actions can be set by this function call

        Attributes:
            actions: a list of strings containing the actions that will be transformed to embeddings using the FeatureEmbeddings
        """
        # Build action embeddings
        self.action_embeddings = []
        self.actions = actions
        action_feat_ind_orig = len(self.embeddings_model.encode(""))
        action_feat_ind = action_feat_ind_orig
        for d in self.actions:
            action_str = d
            action_embed = ""
            for emb in self.embeddings_model.encode(action_str):
                action_embed += f"{action_feat_ind}:{emb} "
                action_feat_ind += 1
            action_feat_ind = action_feat_ind_orig
            self.action_embeddings.append(action_embed)

    def to_vw_example_format(self, context_embed, action_embs, cb_label=None) -> str:
        if cb_label is not None:
            chosen_action, cost, prob = cb_label
        example_string = ""
        example_string += f"shared |Context {context_embed}\n"
        for i, action_embedding in enumerate(action_embs):
            if cb_label is not None and chosen_action == i:
                example_string += "{}:{}:{} ".format(chosen_action, cost, prob)
            example_string += "|Action {} \n".format(action_embedding)
        # Strip the last newline
        return example_string[:-1]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        if not self.actions or not self.action_embeddings:
            raise ValueError("Actions must be set before calling the chain")
        if self.workspace is None:
            raise RuntimeError("Workspace must be set before calling the chain")

        context = inputs[self.context]
        text_parser = vw.TextFormatParser(self.workspace)

        context_embed = ""
        feat = 0
        for emb in self.embeddings_model.encode(context):
            context_embed += f"{feat}:{emb} "
            feat += 1
        # Only supports single example per prompt
        vw_ex = self.to_vw_example_format(context_embed, self.action_embeddings)
        multi_ex = parse_lines(text_parser, vw_ex)
        preds: List[Tuple[int, float]] = self.workspace.predict_one(multi_ex)
        prob_sum = sum(prob for _, prob in preds)
        probabilities = [prob / prob_sum for _, prob in preds]

        ## explore
        sampled_index = np.random.choice(len(preds), p=probabilities)
        sampled_ap = preds[sampled_index]
        sampled_action = sampled_ap[0]
        sampled_prob = sampled_ap[1]

        predicted_action_str = self.actions[sampled_action]

        llm_resp:Dict[str, Any] = super()._call(
            run_manager=run_manager, inputs=inputs, preds=predicted_action_str
        )
        latest_cost = None

        if self.response_checker:
            try:
                cost = -1.0 * self.response_checker.grade_response(
                    inputs=inputs,
                    llm_response=llm_resp[self.output_key],
                    chosen_action=predicted_action_str,
                )

                latest_cost = cost

                text_parser = vw.TextFormatParser(self.workspace)

                cb_label = (sampled_action, cost, sampled_prob)

                vw_ex = self.to_vw_example_format(
                    context_embed, self.action_embeddings, cb_label
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
            context_emb=context_embed,
            action_embeddings=self.action_embeddings,
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

        vw_ex = self.to_vw_example_format(
            response_result.context_emb, response_result.action_embeddings, cb_label
        )
        multi_ex = parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)


# ### TODO:
# - add a callback for them to define the features they want
# - persist data to log file?
# - would this work with a longer chain?
# - make more namespaces available to the user
