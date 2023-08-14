from __future__ import annotations

import logging
import glob
import re
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from collections import defaultdict

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


class _BasedOn:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    __repr__ = __str__


def BasedOn(anything):
    return _BasedOn(anything)


class _ToSelectFrom:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    __repr__ = __str__


def ToSelectFrom(anything):
    if not isinstance(anything, list):
        raise ValueError("ToSelectFrom must be a list to select from")
    return _ToSelectFrom(anything)


class _Embed:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    __repr__ = __str__


def Embed(anything):
    if isinstance(anything, _ToSelectFrom):
        return ToSelectFrom(Embed(anything.value))
    elif isinstance(anything, _BasedOn):
        return BasedOn(Embed(anything.value))
    if isinstance(anything, list):
        return [Embed(v) for v in anything]
    elif isinstance(anything, dict):
        return {k: _Embed(v) for k, v in anything.items()}
    return _Embed(anything)


def parse_lines(parser: vw.TextFormatParser, input_str: str) -> List[vw.Example]:
    return [parser.parse_line(line) for line in input_str.split("\n")]


class Label(ABC):
    pass


class Event(ABC):
    inputs: Dict[str, Any]
    label: Optional[Label]

    def __init__(self, inputs: Dict[str, Any], label: Optional[Label] = None):
        self.inputs = inputs
        self.label = label


class Policy(ABC):
    @abstractmethod
    def predict(self, event: Event) -> Any:
        pass

    @abstractmethod
    def learn(self, event: Event):
        pass

    @abstractmethod
    def log(self, event: Event):
        pass


class VwPolicy(Policy):
    def __init__(
        self,
        workspace: vw.Workspace,
        text_embedder: Embedder,
        logger: VwLogger,
        *_,
        **__,
    ):
        self.workspace = workspace
        self.text_embedder = text_embedder
        self.logger = logger

    def predict(self, event: Event) -> Any:
        text_parser = vw.TextFormatParser(self.workspace)
        return self.workspace.predict_one(
            parse_lines(text_parser, self.text_embedder.to_vw_format(event))
        )

    def learn(self, event: Event):
        vw_ex = self.text_embedder.to_vw_format(event)

        text_parser = vw.TextFormatParser(self.workspace)
        multi_ex = parse_lines(text_parser, vw_ex)
        self.workspace.learn_one(multi_ex)

    def log(self, event: Event):
        vw_ex = self.text_embedder.to_vw_format(event)
        self.logger.log(vw_ex)


class Embedder(ABC):
    @abstractmethod
    def to_vw_format(self, vw_event: Event) -> str:
        pass


class ResponseValidator(ABC):
    """Abstract method to grade the chosen action or the response of the llm"""

    @abstractmethod
    def grade_response(self, inputs: Dict[str, Any], llm_response: str) -> float:
        pass


class RLChain(Chain):
    """
    RLChain class that utilizes the Vowpal Wabbit (VW) model for personalization.

    Attributes:
        model_loading (bool, optional): If set to True, the chain will attempt to load an existing VW model from the latest checkpoint file in the {model_save_dir} directory (current directory if none specified). If set to False, it will start training from scratch, potentially overwriting existing files. Defaults to True.
        large_action_spaces (bool, optional): If set to True and vw_cmd has not been specified in the constructor, it will enable large action spaces
        vw_cmd (List[str], optional): Advanced users can set the VW command line to whatever they want, as long as it is compatible with the Type that is specified (Type Enum)
        model_save_dir (str, optional): The directory to save the VW model to. Defaults to the current directory.
        response_validator (ResponseValidator): If set, the chain will check the response using the provided response_validator and the VW model will be updated with the result. Defaults to None.

    Notes:
        The class creates a VW model instance using the provided arguments. Before the chain object is destroyed the save_progress() function can be called. If it is called, the learned VW model is saved to a file in the current directory named `model-<checkpoint>.vw`. Checkpoints start at 1 and increment monotonically.
        When making predictions, VW is first called to choose action(s) which are then passed into the prompt with the key `{actions}`. After action selection, the LLM (Language Model) is called with the prompt populated by the chosen action(s), and the response is returned.
    """

    llm_chain: Chain
    next_checkpoint: int = 1
    model_save_dir: str = "./"

    output_key: str = "result"  #: :meta private:
    prompt: PromptTemplate
    response_validator: Union[ResponseValidator, None]
    policy: Optional[Policy]

    def __init__(
        self,
        text_embedder: Embedder,
        model_loading=True,
        vw_cmd=[],
        policy=VwPolicy,
        vw_logs: Optional[Union[str, os.PathLike]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.response_validator is None:
            logger.warning(
                "No response validator provided. This is not recommended for RLChains."
            )
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
        workspace = None
        if serialized_workspace:
            workspace = vw.Workspace(vw_cmd, model_data=serialized_workspace)
        else:
            workspace = vw.Workspace(vw_cmd)

        self.policy = policy(
            workspace=workspace, text_embedder=text_embedder, logger=VwLogger(vw_logs)
        )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return []

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def get_context_and_actions(self, inputs: Dict[str, Any]):
        named_actions = {
            k: inputs[k].value
            for k in inputs.keys()
            if isinstance(inputs[k], _ToSelectFrom)
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
            if isinstance(inputs[k], _BasedOn)
        }

        return context, named_actions

    @abstractmethod
    def call_before_predict(self, inputs: Dict[str, Any]) -> Event:
        pass

    @abstractmethod
    def call_after_predict_before_llm(
        self, inputs: Dict[str, Any], event: Event, vwpreds: Any
    ) -> Tuple[Dict[str, Any], Event]:
        pass

    @abstractmethod
    def call_after_llm_before_scoring(
        self, llm_response: str, event: Event
    ) -> Tuple[Dict[str, Any], Event]:
        pass

    @abstractmethod
    def call_after_scoring_before_learning(
        self, llm_response: str, event: Event, response_quality: Optional[float]
    ) -> Event:
        pass

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        event = self.call_before_predict(inputs=inputs)
        vw_pred = self.policy.predict(event=event)

        next_chain_inputs, event = self.call_after_predict_before_llm(
            inputs=inputs, event=event, vwpreds=vw_pred
        )

        t = self.llm_chain.run(**next_chain_inputs, callbacks=_run_manager.get_child())
        _run_manager.on_text(t, color="green", verbose=self.verbose)
        t = t.strip()

        if self.verbose:
            _run_manager.on_text("\nCode: ", verbose=self.verbose)

        output = t
        _run_manager.on_text("\nAnswer: ", verbose=self.verbose)
        _run_manager.on_text(output, color="yellow", verbose=self.verbose)

        next_chain_inputs, event = self.call_after_llm_before_scoring(
            llm_response=output, event=event
        )

        response_quality = None
        try:
            if self.response_validator:
                response_quality = self.response_validator.grade_response(
                    inputs=next_chain_inputs, llm_response=output
                )
        except Exception as e:
            logger.info(
                f"The LLM was not able to rank and the chain was not able to adjust to this response, error: {e}"
            )

        event = self.call_after_scoring_before_learning(
            llm_response=output, response_quality=response_quality, event=event
        )

        self.policy.learn(event=event)
        self.policy.log(event=event)

        return {self.output_key: {"response": output, "response_result": event}}

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


def is_stringtype_instance(item: Any) -> bool:
    """Helper function to check if an item is a string."""
    return isinstance(item, str) or (
        isinstance(item, _Embed) and isinstance(item.value, str)
    )


def embed_string_type(
    item: Union[str, _Embed], model: Any, namespace: Optional[str] = None
) -> Dict[str, str]:
    """Helper function to embed a string or an _Embed object."""
    join_char = ""
    if isinstance(item, _Embed):
        encoded = model.encode(item.value)
        join_char = " "
    elif isinstance(item, str):
        encoded = item
        join_char = ""
    else:
        raise ValueError(f"Unsupported type {type(item)} for embedding")

    if namespace is None:
        raise ValueError(
            "The default namespace must be provided when embedding a string or _Embed object."
        )

    return {namespace: join_char.join(map(str, encoded))}


def embed_dict_type(item: Dict, model: Any) -> Dict[str, Union[str, List[str]]]:
    """Helper function to embed a dictionary item."""
    inner_dict = {}
    for ns, embed_item in item.items():
        if isinstance(embed_item, list):
            inner_dict[ns] = []
            for embed_list_item in embed_item:
                embedded = embed_string_type(embed_list_item, model, ns)
                inner_dict[ns].append(embedded[ns])
        else:
            inner_dict.update(embed_string_type(embed_item, model, ns))
    return inner_dict


def embed_list_type(
    item: list, model: Any, namespace: Optional[str] = None
) -> List[Dict[str, Union[str, List[str]]]]:
    ret_list = []
    for embed_item in item:
        if isinstance(embed_item, dict):
            ret_list.append(embed_dict_type(embed_item, model))
        else:
            ret_list.append(embed_string_type(embed_item, model, namespace))
    return ret_list


def embed(
    to_embed: Union[
        Union(str, _Embed(str)), Dict, List[Union(str, _Embed(str))], List[Dict]
    ],
    model: Any,
    namespace: Optional[str] = None,
) -> List[Dict[str, Union[str, List[str]]]]:
    """
    Embeds the actions or context using the SentenceTransformer model

    Attributes:
        to_embed: (Union[Union(str, _Embed(str)), Dict, List[Union(str, _Embed(str))], List[Dict]], required) The text to be embedded, either a string, a list of strings or a dictionary or a list of dictionaries.
        namespace: (str, optional) The default namespace to use when dictionary or list of dictionaries not provided.
        model: (Any, required) The model to use for embedding
    Returns:
        List[Dict[str, str]]: A list of dictionaries where each dictionary has the namespace as the key and the embedded string as the value
    """
    if (isinstance(to_embed, _Embed) and isinstance(to_embed.value, str)) or isinstance(
        to_embed, str
    ):
        return [embed_string_type(to_embed, model, namespace)]
    elif isinstance(to_embed, dict):
        return [embed_dict_type(to_embed, model)]
    elif isinstance(to_embed, list):
        return embed_list_type(to_embed, model, namespace)
    else:
        raise ValueError("Invalid input format for embedding")
