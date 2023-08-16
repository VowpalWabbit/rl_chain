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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from sentence_transformers import SentenceTransformer


class PickBestTextEmbedder(base.Embedder):
    """
    Contextual Bandit Text Embedder class that embeds the context and actions into a format that can be used by VW
    
    Attributes:
        model name (Any, optional): The type of embeddings to be used for feature representation. Defaults to BERT SentenceTransformer.
    """

    def __init__(self, model: Optional[Any] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if model is None:
            model = SentenceTransformer("bert-base-nli-mean-tokens")

        self.model = model

    def to_vw_format(self, event: PickBest.Event) -> str:
        """
        Converts the context and actions into a format that can be used by VW

        Attributes:
            - cb_label: (Tuple, optional) The tuple containing the chosen action, the cost of the chosen action, and the probability of the chosen action. This tuple is used to label the chosen action in the VW example string.        
            - inputs: (Dict[str, Any]) dictionary containing:
                - context: (Dict) The context for the VW model to use when choosing an action.
                - actions: (List) The list of actions for the VW model to choose from.
        
        Returns:    
        """

        if event.label:
            chosen_action = event.label.chosen_action
            cost = event.label.cost
            prob = event.label.chosen_action_probability

        context_emb = base.embed(event.context, self.model) if event.context else None
        actions_var_name, actions = next(iter(event.actions.items()), (None, None))
        action_embs = (
            base.embed(actions, self.model, actions_var_name) if event.actions else None
        )

        if not context_emb or not action_embs:
            raise ValueError(
                "Context and actions must be provided in the inputs dictionary"
            )

        example_string = ""
        example_string += f"shared "
        for context_item in context_emb:
            for ns, context in context_item.items():
                example_string += f"|{ns} {' '.join(context) if isinstance(context, list) else context} "
        example_string += "\n"

        for i, action in enumerate(action_embs):
            if event.label and event.label.cost is not None and chosen_action == i:
                example_string += f"{chosen_action}:{cost}:{prob} "
            for ns, action_embedding in action.items():
                example_string += f"|{ns} {' '.join(action_embedding) if isinstance(action_embedding, list) else action_embedding} "
            example_string += "\n"
        # Strip the last newline
        return example_string[:-1]


class PickBestAutoResponseValidator(base.ResponseValidator):
    llm_chain: LLMChain
    prompt: PromptTemplate
    default_system_prompt = SystemMessagePromptTemplate.from_template(
        "PLEASE RESPOND ONLY WITH A SIGNLE FLOAT AND NO OTHER TEXT EXPLANATION\n You are a strict judge that is called on to rank a response based on given criteria.\
                You must respond with your ranking by providing a single float within the range [-1, 1], -1 being very bad response and 1 being very good response."
    )

    def __init__(self, llm, prompt=None):
        if prompt:
            self.prompt = prompt
        else:
            human_template = 'Given this context "{best_pick_context}" as the most important attribute, rank how good or bad this text selection is: "{best_pick}".'
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                human_template
            )

            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    PickBestAutoResponseValidator.default_system_prompt,
                    human_message_prompt,
                ]
            )
            self.prompt = chat_prompt

        self.llm_chain = LLMChain(llm=llm, prompt=self.prompt)

    def grade_response(self, inputs: Dict[str, Any], llm_response: str) -> float:
        ranking = self.llm_chain.predict(llm_response=llm_response, **inputs)
        ranking = ranking.strip()
        try:
            resp = float(ranking)
            return resp
        except Exception as e:
            raise RuntimeError(
                f"The llm did not manage to rank the response as expected, there is always the option to try again or tweak the reward prompt. Error: {e}"
            )


class PickBest(base.RLChain):
    """
    PickBest class that utilizes the Vowpal Wabbit (VW) model for personalization.

    The Chain is initialized with a set of potential actions. For each call to the Chain, a specific action will be chosen based on an input context.
    This chosen action is then passed to the prompt that will be utilized in the subsequent call to the LLM (Language Model).

    The flow of this chain is:
    - Chain is initialized
    - Chain is called input containing the context and the List of potential actions
    - Chain chooses an action based on the context
    - Chain calls the LLM with the chosen action
    - LLM returns a response
    - If the response_validator is specified, the response is checked against the response_validator
    - The internal model will be updated with the context, action, and reward of the response (how good or bad the response was)
    - The response is returned

    input dictionary expects:
        - at least one variable wrapped in BasedOn which will be the context to use for personalization
        - one variable of a list wrapped in ToSelectFrom which will be the list of actions for the Vowpal Wabbit model to choose from.
            This list can either be a List of str's or a List of Dict's.
                - Actions provided as a list of strings e.g. actions = ["action1", "action2", "action3"]
                - If actions are provided as a list of dictionaries, each action should be a dictionary where the keys are namespace names and the values are the corresponding action strings e.g. actions = [{"namespace1": "action1", "namespace2": "action2"}, {"namespace1": "action3", "namespace2": "action4"}]
    Extends:
        RLChain

    Attributes:
        text_embedder: (PickBestTextEmbedder, optional) The text embedder to use for embedding the context and the actions. If not provided, a default embedder is used.
    """

    class Label(base.Label):
        chosen_action: Optional[int]
        chosen_action_probability: Optional[float]
        cost: Optional[float]

        def __init__(
            self,
            chosen_action: Optional[int] = None,
            chosen_action_probability: Optional[float] = None,
            cost: Optional[float] = None,
        ):
            self.chosen_action = chosen_action
            self.chosen_action_probability = chosen_action_probability
            self.cost = cost

    class Event(base.Event):
        def __init__(
            self,
            inputs: Dict[str, Any],
            actions: Dict[str, Any],
            context: Dict[str, Any],
            label: Optional[PickBest.Label] = None,
        ):
            super().__init__(inputs=inputs, label=label)
            self.actions = actions
            self.context = context

    best_pick_input_key = "best_pick"
    best_pick_context_input_key = "best_pick_context"

    def __init__(
        self, text_embedder: Optional[PickBestTextEmbedder] = None, *args, **kwargs
    ):
        vw_cmd = kwargs.get("vw_cmd", [])
        if not vw_cmd:
            vw_cmd = [
                "--cb_explore_adf",
                "--quiet",
                "--interactions=::",
                "--coin",
                "--epsilon=0.2",
            ]
        else:
            if "--cb_explore_adf" not in vw_cmd:
                raise ValueError(
                    "If vw_cmd is specified, it must include --cb_explore_adf"
                )

        kwargs["vw_cmd"] = vw_cmd
        if not text_embedder:
            text_embedder = PickBestTextEmbedder()

        super().__init__(text_embedder=text_embedder, *args, **kwargs)

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        super()._validate_inputs(inputs)
        if (
            self.best_pick_input_key in inputs.keys()
            or self.best_pick_context_input_key in inputs.keys()
        ):
            raise ValueError(
                f"The PickBest chain does not accept '{self.best_pick_input_key}' or '{self.best_pick_context_input_key}' as input keys, they are reserved for internal use during auto reward."
            )

    def _call_before_predict(self, inputs: Dict[str, Any]) -> PickBest.Event:
        context, actions = base.get_context_and_actions(inputs=inputs)
        if not actions:
            raise ValueError(
                "No variables using 'ToSelectFrom' found in the inputs. Please include at least one variable containing a list to select from."
            )

        if len(list(actions.values())) > 1:
            raise ValueError(
                "Only one variable using 'ToSelectFrom' can be provided in the inputs for the PickBest chain. Please provide only one variable containing a list to select from."
            )

        if not context:
            raise ValueError(
                "No variables using 'BasedOn' found in the inputs. Please include at least one variable containing information to base the selection of ToSelectFrom on."
            )

        event = PickBest.Event(inputs=inputs, actions=actions, context=context)
        return event

    def _call_after_predict_before_llm(
        self, inputs: Dict[str, Any], event: Event, prediction: List[Tuple[int, float]]
    ) -> Tuple[Dict[str, Any], PickBest.Event]:
        prob_sum = sum(prob for _, prob in prediction)
        probabilities = [prob / prob_sum for _, prob in prediction]
        ## sample from the pmf
        sampled_index = np.random.choice(len(prediction), p=probabilities)
        sampled_ap = prediction[sampled_index]
        sampled_action = sampled_ap[0]
        sampled_prob = sampled_ap[1]
        label = PickBest.Label(
            chosen_action=sampled_action, chosen_action_probability=sampled_prob
        )
        event.label = label

        # only one key, value pair in event.actions
        key, value = next(iter(event.actions.items()))
        next_chain_inputs = inputs.copy()
        next_chain_inputs.update({key: value[event.label.chosen_action]})
        return next_chain_inputs, event

    def _call_after_llm_before_scoring(
        self, llm_response: str, event: PickBest.Event
    ) -> Tuple[Dict[str, Any], PickBest.Event]:
        next_chain_inputs = event.inputs.copy()
        # only one key, value pair in event.actions
        value = next(iter(event.actions.values()))
        next_chain_inputs.update(
            {
                self.best_pick_context_input_key: str(event.context),
                self.best_pick_input_key: value[event.label.chosen_action],
            }
        )
        return next_chain_inputs, event

    def _call_after_scoring_before_learning(
        self, event: PickBest.Event, response_quality: Optional[float]
    ) -> Event:
        event.label.cost = (
            -1.0 * response_quality if response_quality is not None else None
        )
        return event

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        When chain.run() is called with the given inputs, this function is called. It is responsible for calling the VW model to choose an action (ToSelectFrom) based on the (BasedOn) context, and then calling the LLM (Language Model) with the chosen action to generate a response.

        Attributes:
            inputs: (Dict, required) The inputs to the chain. The inputs must contain a input variables that are wrapped in BasedOn and ToSelectFrom. BasedOn is the context that will be used for selecting an ToSelectFrom action that will be passed to the LLM prompt.
            run_manager: (CallbackManagerForChainRun, optional) The callback manager to use for this run. If not provided, a default callback manager is used.
            
        Returns:
            A dictionary containing:
                - `response`: The response generated by the LLM (Language Model).
                - `decision_metadata`: A Event object containing all the information needed to learn the reward for the chosen action at a later point. If an automatic response_validator is not provided, then this object can be used at a later point with the `learn_delayed_reward()` function to learn the delayed reward and update the Vowpal Wabbit model.
                    - the `cost` in the `decision_metadata` object is set to None if an automatic response_validator is not provided or if the response_validator failed (e.g. LLM timeout or LLM failed to rank correctly).
        """
        return super()._call(run_manager=run_manager, inputs=inputs)

    @property
    def _chain_type(self) -> str:
        return "llm_rl_chain_pick_best_chain"

    @classmethod
    def from_chain(cls, llm_chain: Chain, prompt: PromptTemplate, **kwargs: Any):
        return PickBest(llm_chain=llm_chain, prompt=prompt, **kwargs)

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt: PromptTemplate, **kwargs: Any):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return PickBest.from_chain(llm_chain=llm_chain, prompt=prompt, **kwargs)
