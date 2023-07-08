"""Chain that interprets a prompt and executes bash code to perform bash operations."""
from __future__ import annotations

import logging
import glob
import re
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer
import vowpal_wabbit_next as vw
from .personalizer_prompt import PROMPT

from pydantic import Extra, Field, root_validator
import numpy as np

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import OutputParserException
from enum import Enum

logger = logging.getLogger(__name__)


def parse_lines(parser: vw.TextFormatParser, input_str: str) -> List[vw.Example]:
    return [parser.parse_line(line) for line in input_str.split("\n")]


class PersonalizerChain(Chain):
    """
    PersonalizerChain class that utilizes the Vowpal Wabbit (VW) model for personalization.

    Attributes:
        vw_workspace_type (Type): The type of personalization algorithm to be used by the VW model.
        embeddings (FeatureEmbeddings, optional): The type of embeddings to be used for feature representation. Defaults to BERT.
        model_loading (bool, optional): If set to True, the chain will attempt to load an existing VW model from the latest checkpoint file in the {model_save_dir} directory (current directory if none specified). If set to False, it will start training from scratch, potentially overwriting existing files. Defaults to True.
        actions (List[str], optional): A list of action strings for VW to choose from.
        large_action_spaces (bool, optional): If set to True and vw_cmd has not been specified in the constructor, it will enable large action spaces
        vw_cmd (List[str], optional): Advanced users can set the VW command line to whatever they want, as long as it is compatible with the Type that is specified (Type Enum)
        model_save_dir (str, optional): The directory to save the VW model to. Defaults to the current directory.

    Notes:
        The class creates a VW model instance using the provided arguments. Before the chain object is destroyed the save_progress() function can be called. If it is called, the learned VW model is saved to a file in the current directory named `model-<checkpoint>.vw`. Checkpoints start at 1 and increment monotonically.
        When making predictions, VW is first called to choose action(s) which are then passed into the prompt with the key `{actions}`. After action selection, the LLM (Language Model) is called with the prompt populated by the chosen action(s), and the response is returned.
    """

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None

    workspace: vw.Workspace = None
    sbert_model: SentenceTransformer = None

    action_embeddings: List = []
    actions: List = []

    next_checkpoint: int = None

    model_save_dir: str = "./"

    class Type(Enum):
        """
        Enumeration to define the type of personalization algorithm to be used in the VW model.

        Attributes:
            CONTEXTUAL_BANDITS (tuple): Indicates the use of the Contextual Bandits algorithm.
            CONDITIONAL_CONTEXTUAL_BANDITS (tuple): Indicates the use of the Conditional Contextual Bandits algorithm.
        """

        CONTEXTUAL_BANDITS = (1,)
        CONDITIONAL_CONTEXTUAL_BANDITS = (2,)

    class FeatureEmbeddings(Enum):
        """
        Enumeration to define the type of embeddings used to featurize the string context and actions in the VW model.

        Note:
            The use of a different embedding type with a pre-existing model may result in undefined behavior.

        Attributes:
            BERT (int): Indicates the use of BERT embeddings.
        """

        BERT = 2

    def __init__(
        self,
        vw_workspace_type: Type,
        embeddings=FeatureEmbeddings.BERT,
        model_loading=True,
        large_action_spaces=False,
        vw_cmd=[],
        actions: List[str] = [],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        next_checkpoint = 1
        serialized_workspace = None

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
        print(f"next checkpoint = {self.next_checkpoint}")

        las_cmd = []
        if large_action_spaces and not vw_cmd:
            las_cmd = ["--large_action_space"]

        if vw_workspace_type == PersonalizerChain.Type.CONTEXTUAL_BANDITS:
            if not vw_cmd:
                vw_cmd = las_cmd + [
                    "--cb_explore_adf",
                    "--quiet",
                    "--interactions=AC",
                    "--coin",
                    "--squarecb",
                ]
        elif vw_workspace_type == PersonalizerChain.Type.CONDITIONAL_CONTEXTUAL_BANDITS:
            if not vw_cmd:
                vw_cmd = las_cmd + [
                    "--ccb_explore_adf",
                    "--quiet",
                    "--interactions=AC",
                    "--coin",
                    "--squarecb",
                ]
        else:
            raise ValueError("No other vw types supported yet")

        if embeddings == PersonalizerChain.FeatureEmbeddings.BERT:
            self.sbert_model = SentenceTransformer("bert-base-nli-mean-tokens")
        else:
            raise ValueError("No other sentence transformers supported yet")

        print(f"vw command: {vw_cmd}")
        # initialize things
        if actions:
            self.set_actions(actions)
        if serialized_workspace:
            self.workspace = vw.Workspace(vw_cmd, model_data=serialized_workspace)
        else:
            self.workspace = vw.Workspace(vw_cmd)

    context: str = "context"  #: :meta private:

    output_key: str = "answer"  #: :meta private:
    prompt: BasePromptTemplate = PROMPT

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
        action_feat_ind_orig = len(self.sbert_model.encode(""))
        action_feat_ind = action_feat_ind_orig
        for d in actions:
            action_str = d
            action_embed = ""
            for emb in self.sbert_model.encode(action_str):
                action_embed += f"{action_feat_ind}:{emb} "
                action_feat_ind += 1
            action_feat_ind = action_feat_ind_orig
            self.action_embeddings.append(action_embed)

    def _call(
        self,
        inputs: Dict[str, Any],
        preds: Any,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        # print(f"THE PREDS I GOT: {preds}")

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
        print(f"storing in: {self.model_save_dir}/model-{self.next_checkpoint}.vw")
        with open(f"{self.model_save_dir}/model-{self.next_checkpoint}.vw", "wb") as f:
            f.write(serialized_workspace)

    @property
    def _chain_type(self) -> str:
        return "llm_personalizer_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PROMPT,
        **kwargs: Any,
    ) -> ContextualBanditPersonalizerChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        if "vw_workspace_type" not in kwargs:
            raise ValueError("vw_workspace_type must be specified")
        if kwargs.get("vw_workspace_type") is PersonalizerChain.Type.CONTEXTUAL_BANDITS:
            return ContextualBanditPersonalizerChain(llm_chain=llm_chain, **kwargs)
        elif (
            kwargs.get("vw_workspace_type")
            is PersonalizerChain.Type.CONDITIONAL_CONTEXTUAL_BANDITS
        ):
            raise ValueError("Not implemented yet")
        else:
            raise ValueError("Type not supported")


class ContextualBanditPersonalizerChain(PersonalizerChain):
    latest_context_emb: List = None
    latest_prob: float = None
    latest_action: int = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_vw_example_format(self, context_embed, actions, cb_label=None) -> str:
        if cb_label is not None:
            chosen_action, cost, prob = cb_label
        example_string = ""
        example_string += f"shared |Context {context_embed}\n"
        for i, action in enumerate(actions):
            if cb_label is not None and chosen_action == i:
                example_string += "{}:{}:{} ".format(chosen_action, cost, prob)
            example_string += "|Action {} \n".format(action)
        # Strip the last newline
        return example_string[:-1]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        context = inputs[self.context]
        text_parser = vw.TextFormatParser(self.workspace)

        context_embed = ""
        feat = 0
        for emb in self.sbert_model.encode(context):
            context_embed += f"{feat}:{emb} "
            feat += 1
        # Only supports single example per prompt
        vw_ex = self.to_vw_example_format(context_embed, self.action_embeddings)
        multi_ex = parse_lines(text_parser, vw_ex)
        self.latest_context_emb = context_embed
        preds = self.workspace.predict_one(multi_ex)

        # print(f"preds: {preds}")
        prob_sum = sum(prob for _, prob in preds)
        probabilities = [prob / prob_sum for _, prob in preds]

        ## explore
        sampled_index = np.random.choice(len(preds), p=probabilities)
        sampled_ap = preds[sampled_index]
        ## exploit
        # sampled_ap = max(preds, key=lambda x : x[1])

        sampled_action = sampled_ap[0]
        self.latest_action = sampled_action
        self.latest_prob = sampled_ap[1]

        winer_text = self.actions[sampled_action]

        return super()._call(run_manager=run_manager, inputs=inputs, preds=winer_text)

    def good_recommendation(self):
        """
        Learn will be called with a cost of -1 (cost range if good/neutural/bad are used is [-1, 1])

        Note:
            Care is needed if mixed with calling learn_with_specific_cost
        """
        text_parser = vw.TextFormatParser(self.workspace)

        # reward means the smallest cost
        cb_label = (self.latest_action, -1, self.latest_prob)
        vw_ex = self.to_vw_example_format(
            self.latest_context_emb, self.action_embeddings, cb_label
        )
        multi_ex = parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)

    def neutral_recommendation(self):
        """
        Learn will be called with a cost of 0 (cost range if good/neutural/bad are used is [-1, 1])

        Note:
            Care is needed if mixed with calling learn_with_specific_cost
        """
        text_parser = vw.TextFormatParser(self.workspace)

        # punish means the intermediate cost
        cb_label = (self.latest_action, 0, self.latest_prob)

        vw_ex = self.to_vw_example_format(
            self.latest_context_emb, self.action_embeddings, cb_label
        )
        multi_ex = parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)

    def bad_recommendation(self):
        """
        Learn will be called with a cost of 1 (cost range if good/neutural/bad are used is [-1, 1])

        Note:
            Care is needed if mixed with calling learn_with_specific_cost
        """
        text_parser = vw.TextFormatParser(self.workspace)

        # punish means the biggest cost
        cb_label = (self.latest_action, 1, self.latest_prob)

        vw_ex = self.to_vw_example_format(
            self.latest_context_emb, self.action_embeddings, cb_label
        )
        multi_ex = parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)

    def learn_with_specific_cost(self, cost: int):
        """
        Learn will be called with the cost specified

        Note:
            Care is needed if mixed with calling good/bad/neutural_reccommendation methods
        """
        text_parser = vw.TextFormatParser(self.workspace)

        cb_label = (self.latest_action, cost, self.latest_prob)

        vw_ex = self.to_vw_example_format(
            self.latest_context_emb, self.action_embeddings, cb_label
        )
        multi_ex = parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)


# ### TODO:
# - simple joining
# - add a callback for them to define the features they want
