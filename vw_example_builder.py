from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from sentence_transformers import SentenceTransformer


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
    def to_vw_format(self, **kwargs):
        pass


class ContextualBanditTextEmbedder(Embedder):
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
        return embed(actions, self.model, "Actions")

    def embed_context(self, context: Any):
        """
        Embeds the context using the SentenceTransformer model

        Attributes:
            context: (Any, required) The context for the VW model to use when choosing an action. This can either be a str or a Dict.
            - If context is provided as a string (e.g. "context"), the context will be assigned to the Vowpal Wabbit namespace, labelled `Context`.
            - If context is provided as a dictionary, then it should be a single dictionary where the keys are namespace names and the values are the corresponding strings of the context (e.g. {"namespace1": "part of context", "namespace2": "another part of the context"})
        """
        return embed(context, self.model, "Context")[0]

    def to_vw_format(self, **kwargs):
        """
        Converts the context and actions into a format that can be used by VW

        Attributes:
            **kwargs: (Dict, optional) The keyword arguments that can be passed to the function. The following keyword arguments are supported:
                - cb_label: (Tuple, optional) The tuple containing the chosen action, the cost of the chosen action, and the probability of the chosen action. This tuple is used to label the chosen action in the VW example string.        
                - inputs: (Dict[str, Any], optional) dictionary containing:
                    - context: (Dict, optional) The context for the VW model to use when choosing an action.
                    - actions: (List, optional) The list of actions for the VW model to choose from.
        
        Returns:    
        """

        if "cb_label" in kwargs:
            chosen_action, cost, prob = kwargs["cb_label"]
        
        inputs = kwargs.get('inputs', {})
        context = inputs.get('context')
        actions = inputs.get('actions')

        context_emb = self.embed_context(context) if context else None
        action_embs = self.embed_actions(actions) if actions else None

        if not context_emb or not action_embs:
            raise ValueError("Context and actions must be provided in the inputs dictionary")

        example_string = ""
        example_string += f"shared "
        for ns, context in context_emb.items():
            example_string += f"|{ns} {context} "
        example_string += "\n"

        for i, action in enumerate(action_embs):
            if "cb_label" in kwargs and chosen_action == i:
                example_string += f"{chosen_action}:{cost}:{prob} "
            for ns, action_embedding in action.items():
                example_string += f"|{ns} {action_embedding} "
            example_string += "\n"
        # Strip the last newline
        return example_string[:-1]