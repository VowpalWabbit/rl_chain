from __future__ import annotations

import logging
import glob
import re
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import vowpal_wabbit_next as vw
from .vw_logger import VwLogger
from langchain.prompts.prompt import PromptTemplate

from pydantic import Extra, PrivateAttr

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
logger.addHandler(ch)

def parse_lines(parser: vw.TextFormatParser, input_str: str) -> List[vw.Example]:
    return [parser.parse_line(line) for line in input_str.split("\n")]

class RLChain(Chain):
    """
    RLChain class that utilizes the Vowpal Wabbit (VW) model for personalization.

    Attributes:
        model_loading (bool, optional): If set to True, the chain will attempt to load an existing VW model from the latest checkpoint file in the {model_save_dir} directory (current directory if none specified). If set to False, it will start training from scratch, potentially overwriting existing files. Defaults to True.
        large_action_spaces (bool, optional): If set to True and vw_cmd has not been specified in the constructor, it will enable large action spaces
        vw_cmd (List[str], optional): Advanced users can set the VW command line to whatever they want, as long as it is compatible with the Type that is specified (Type Enum)
        model_save_dir (str, optional): The directory to save the VW model to. Defaults to the current directory.
        response_validator (ResponseValidator, optional): If set, the chain will check the response using the provided response_validator and the VW model will be updated with the result. Defaults to None.

    Notes:
        The class creates a VW model instance using the provided arguments. Before the chain object is destroyed the save_progress() function can be called. If it is called, the learned VW model is saved to a file in the current directory named `model-<checkpoint>.vw`. Checkpoints start at 1 and increment monotonically.
        When making predictions, VW is first called to choose action(s) which are then passed into the prompt with the key `{actions}`. After action selection, the LLM (Language Model) is called with the prompt populated by the chosen action(s), and the response is returned.
    """

    llm_chain: Chain
    workspace: Optional[vw.Workspace] = None
    next_checkpoint: int = 1
    model_save_dir: str = "./"
    response_validator: Optional[ResponseValidator] = None
    vw_logger: VwLogger = None

    context: str = "context"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    prompt: PromptTemplate

    def __init__(
        self,
        model_loading=True,
        vw_cmd=[],
        vw_logs: Optional[Union[str, os.PathLike]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vw_logger = VwLogger(vw_logs)
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
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        t = self.llm_chain.run(
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

    def save_progress(self) -> None:
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

    def _learn(self, vw_ex):
        self.vw_logger.log(vw_ex)
        text_parser = vw.TextFormatParser(self.workspace)
        multi_ex = parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)


def embed(
    to_embed: Union[str, Dict, List[str], List[Dict]],
    model: Any,
    default_namespace: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Embeds the actions or context using the SentenceTransformer model

    Attributes:
        to_embed: (Union[str, List[str], List[Dict]], required) The text to be embedded, either a string, a list of strings or a list of dictionaries.
        - if to_embed provided as a string, it will be embedded as a single string with the default namespace as dictionary key
        - if to_embed provided as a dictionary, it will be embedded as a single string
        - if to_embed provided as a list of strings, each string will be embedded as a single string with the default namespace as dictionary keys
        - if to_embed provided as a list of dictionaries, each dictionary will be embedded as a single string
        default_namespace: (str, required) The default namespace to use when embedding the to_embed string
        model: (Any, required) The model to use for embedding
    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary has the namespace as the key and the embedded string as the value
    """
    if isinstance(to_embed, str):
        if default_namespace is None:
            raise ValueError(
                "The default namespace must be provided when embedding a string"
            )
        return [{default_namespace: " ".join(map(str, model.encode(to_embed)))}]
    elif isinstance(to_embed, dict):
        return [
            {
                ns: " ".join(map(str, model.encode(embed_str)))
                for ns, embed_str in to_embed.items()
            }
        ]
    elif isinstance(to_embed, list):
        if isinstance(to_embed[0], str):
            if default_namespace is None:
                raise ValueError(
                    "The default namespace must be provided when embedding a list of strings"
                )
            return [
                {default_namespace: " ".join(map(str, model.encode(embed_item)))}
                for embed_item in to_embed
            ]
        else:
            return [
                {
                    ns: " ".join(map(str, model.encode(embed_str)))
                    for ns, embed_str in embed_item.items()
                }
                for embed_item in to_embed
            ]
    else:
        raise ValueError("Invalid input format for embedding")


class Embedder(ABC):
    @abstractmethod
    def to_vw_format(self, **kwargs) -> str:
        pass

class ResponseValidator(ABC):
    """Abstract method to grade the chosen action or the response of the llm"""

    @abstractmethod
    def grade_response(
        self, inputs: Dict[str, Any], llm_response: str, **kwargs
    ) -> float:
        pass