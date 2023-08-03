from . import rl_chain_base as base 

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.prompts.prompt import PromptTemplate

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
import vowpal_wabbit_next as vw
from typing import Any, Dict, List, Optional, Tuple, Union

from .pick_best_prompt import PROMPT
import numpy as np
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from sentence_transformers import SentenceTransformer

class ContextualBanditTextEmbedder(base.Embedder):
    """
    Contextual Bandit Text Embedder class that embeds the context and actions into a format that can be used by VW
    
    Attributes:
        embeddings_model name (SentenceTransformer, optional): The type of embeddings to be used for feature representation. Defaults to BERT.
    """

    def __init__(self, model_name: Optional[str] = None):
        if not model_name:
            self.model = None
        else:
            self.model = SentenceTransformer(model_name)

    def embed_actions(self, actions: List):
        """
        Embeds the actions using the SentenceTransformer model
        
        Attributes:
            actions: (List, required) The list of actions for the VW model to choose from. This list can either be a List of str's or a List of Dict's.
            - If actions are provided as a list of strings (e.g. actions = ["action1", "action2", "action3"]), each action will be assigned to the Vowpal Wabbit namespace, labelled `Actions`.
            - If actions are provided as a list of dictionaries, each action should be a dictionary where the keys are namespace names and the values are the corresponding action strings (e.g. actions = [{"namespace1": "action1", "namespace2": "action2"}, {"namespace1": "action3", "namespace2": "action4"}])
        """
        return base.embed(actions, self.model, "Actions")

    def embed_context(self, context: Any):
        """
        Embeds the context using the SentenceTransformer model

        Attributes:
            context: (Any, required) The context for the VW model to use when choosing an action. This can either be a str or a Dict.
            - If context is provided as a string (e.g. "context"), the context will be assigned to the Vowpal Wabbit namespace, labelled `Context`.
            - If context is provided as a dictionary, then it should be a single dictionary where the keys are namespace names and the values are the corresponding strings of the context (e.g. {"namespace1": "part of context", "namespace2": "another part of the context"})
        """
        return base.embed(context, self.model, "Context")

    def to_vw_format(self, inputs :Dict[str, Any], cb_label: Optional[Tuple]=None) -> str:
        """
        Converts the context and actions into a format that can be used by VW

        Attributes:
            - cb_label: (Tuple, optional) The tuple containing the chosen action, the cost of the chosen action, and the probability of the chosen action. This tuple is used to label the chosen action in the VW example string.        
            - inputs: (Dict[str, Any]) dictionary containing:
                - context: (Dict) The context for the VW model to use when choosing an action.
                - actions: (List) The list of actions for the VW model to choose from.
        
        Returns:    
        """

        if cb_label:
            chosen_action, cost, prob = cb_label

        context = inputs.get('context')
        actions = inputs.get('actions')

        context_emb = self.embed_context(context) if context else None
        action_embs = self.embed_actions(actions) if actions else None

        if not context_emb or not action_embs:
            raise ValueError("Context and actions must be provided in the inputs dictionary")

        example_string = ""
        example_string += f"shared "
        for context_item in context_emb:
            for ns, context in context_item.items():
                example_string += f"|{ns} {' '.join(context)} "
                # if isinstance(context, str):
                #     example_string += f"|{ns} {context} "
                # elif isinstance(context, list):
                #     example_string += f"|{ns} {' '.join(context)} "
        example_string += "\n"

        for i, action in enumerate(action_embs):
            if cb_label and chosen_action == i:
                example_string += f"{chosen_action}:{cost}:{prob} "
            for ns, action_embedding in action.items():
                example_string += f"|{ns} {action_embedding} "
            example_string += "\n"
        # Strip the last newline
        return example_string[:-1]

class AutoValidatePickBest(base.ResponseValidator):
    llm_chain: LLMChain
    prompt: PromptTemplate

    def __init__(self, llm, prompt=None):
        if prompt:
            self.prompt = prompt
        else:
            template = "PLEASE RESPOND ONLY WITH A SIGNLE FLOAT AND NO OTHER TEXT EXPLANATION\n You are a strict judge that is called on to rank a response based on given criteria.\
                You must respond with your ranking by providing a single float within the range [-1, 1], -1 being very bad response and 1 being very good response."
            system_message_prompt = SystemMessagePromptTemplate.from_template(template)
            human_template = "Given this context \"{context}\" as the most important attribute, rank how good or bad this text selection is: \"{selected}\"."
            human_message_prompt = HumanMessagePromptTemplate.from_template(
                human_template
            )

            chat_prompt = ChatPromptTemplate.from_messages(
                [system_message_prompt, human_message_prompt]
            )
            self.prompt = chat_prompt

        self.llm_chain = LLMChain(llm=llm, prompt=self.prompt)

    def grade_response(self, inputs: Dict[str, Any], llm_response: str, **kwargs) -> float:
        inputs["llm_response"] = llm_response
        inputs["selected"] = inputs["selected"]
        ranking = self.llm_chain.predict(**inputs)
        ranking = ranking.strip()
        try:
            resp = float(ranking)
            return resp
        except Exception:
            raise RuntimeError(
                "The llm did not manage to rank the response as expected, there is always the option to try again"
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
        - context: (str, required) The context to use for personalization
        - actions: (List, required) The list of actions for the Vowpal Wabbit model to choose from. This list can either be a List of str's or a List of Dict's.
                - Actions provided as a list of strings e.g. actions = ["action1", "action2", "action3"]
                - If actions are provided as a list of dictionaries, each action should be a dictionary where the keys are namespace names and the values are the corresponding action strings e.g. actions = [{"namespace1": "action1", "namespace2": "action2"}, {"namespace1": "action3", "namespace2": "action4"}]    
    Extends:
        RLChain

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
                - `response_result`: A ResponseResult object containing all the information needed to learn the reward for the chosen action at a later point. If an automatic response_validator is not provided, then this object can be used at a later point with the `learn_delayed_reward()` function to learn the delayed reward and update the Vowpal Wabbit model.
                    - the `cost` in the `response_result` object is set to None if an automatic response_validator is not provided or if the response_validator failed (e.g. LLM timeout or LLM failed to rank correctly).
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        if self.workspace is None:
            raise RuntimeError("Workspace must be set before calling the chain")

        text_parser = vw.TextFormatParser(self.workspace)

        actions = inputs[self.actions]
        vw_ex = self.text_embedder.to_vw_format(inputs=inputs)
        multi_ex = base.parse_lines(text_parser, vw_ex)
        preds: List[Tuple[int, float]] = self.workspace.predict_one(multi_ex)
        prob_sum = sum(prob for _, prob in preds)
        probabilities = [prob / prob_sum for _, prob in preds]

        ## explore
        sampled_index = np.random.choice(len(preds), p=probabilities)
        sampled_ap = preds[sampled_index]
        sampled_action = sampled_ap[0]
        sampled_prob = sampled_ap[1]

        pred_action = actions[sampled_action]
        inputs['selected'] = pred_action

        llm_resp: Dict[str, Any] = super()._call(run_manager=run_manager, inputs=inputs)

        # llm_resp: Dict[str, Any] = {self.output_key : ""}

        latest_cost = None

        if self.response_validator:
            try:
                cost = -1.0 * self.response_validator.grade_response(
                    inputs=inputs,
                    llm_response=llm_resp[self.output_key]                )
                latest_cost = cost
                cb_label = (sampled_action, cost, sampled_prob)

                vw_ex = self.text_embedder.to_vw_format(
                    cb_label=cb_label,
                    inputs=inputs,
                )
                self._learn(vw_ex)

            except Exception as e:
                base.logger.info(
                    f"The LLM was not able to rank and the chain was not able to adjust to this response. Error: {e}"
                )

        self.latest_response = PickBest.ResponseResult(
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

    @property
    def _chain_type(self) -> str:
        return "llm_personalizer_chain"

    @classmethod
    def from_chain(cls, llm_chain: Chain, prompt: PromptTemplate = PROMPT, **kwargs: Any):
        return PickBest(
            llm_chain=llm_chain, prompt=prompt, **kwargs
        )

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, prompt: PromptTemplate = PROMPT, **kwargs: Any):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return PickBest.from_chain(llm_chain=llm_chain, prompt=prompt, **kwargs)

    def learn_delayed_reward(
        self, reward: float, response_result: ResponseResult, force_reward=False
    ) -> None:
        """
        Learn will be called with the reward specified and the actions/embeddings/etc stored in response_result

        Will raise an error if check_response is set to True and force_reward=True was not provided during the method call
        force_cost should be used if the check_response failed to check the response correctly
        """
        if self.response_validator and not force_reward:
            raise RuntimeError(
                "check_response is set to True, this must be turned off for explicit feedback and training to be provided, or overriden by calling the method with force_reward=True"
            )
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
        self._learn(vw_ex)