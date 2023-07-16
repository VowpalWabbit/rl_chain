from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer


class Embedder(ABC):
    @abstractmethod
    def to_vw_format(self, **kwargs) -> str:
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
        if actions and isinstance(actions[0], str):
            self.actions: List[Dict[str, str]] = []
            for action in actions:
                self.actions.append({"Action": action})
        else:
            self.actions = actions

        self.action_embeddings: List[Dict[str, str]] = []
        for action in self.actions:
            action_dict = {}
            for ns, action_str in action.items():
                action_embed = " ".join(map(str, self.model.encode(action_str)))
                action_dict[ns] = action_embed
            self.action_embeddings.append(action_dict)

    def embed_context(self, context: Any):
        """
        Embeds the context using the SentenceTransformer model

        Attributes:
            context: (Any, required) The context for the VW model to use when choosing an action. This can either be a str or a Dict.
            - If context is provided as a string (e.g. "context"), the context will be assigned to the Vowpal Wabbit namespace, labelled `Context`.
            - If context is provided as a dictionary, then it should be a single dictionary where the keys are namespace names and the values are the corresponding strings of the context (e.g. {"namespace1": "part of context", "namespace2": "another part of the context"})
        """
        if isinstance(context, str):
            self.context = {"Context": context}
        elif isinstance(context, dict):
            self.context = context
        else:
            raise ValueError("Context must be a string or a dictionary")

        self.context_embedding: Dict[str, str] = {}
        for ns, context_str in self.context.items():
            self.context_embedding[ns] = " ".join(map(str, self.model.encode(context_str)))

    def get_context_embedding(self) -> Dict[str, str]:
        return self.context_embedding
    
    def get_action_embeddings(self) -> List[Dict[str, str]]:
        return self.action_embeddings

    def to_vw_format(self, **kwargs) -> str:
        """
        Converts the context and actions into a format that can be used by VW

        Attributes:
            **kwargs: (Dict, optional) The keyword arguments that can be passed to the function. The following keyword arguments are supported:
                - cb_label: (Tuple, optional) The tuple containing the chosen action, the cost of the chosen action, and the probability of the chosen action. This tuple is used to label the chosen action in the VW example string.        
                - context: (Dict, optional) The context for the VW model to use when choosing an action. If not supplied then the internally stored context from the latest embed_context() call will be used.
                - actions: (List, optional) The list of actions for the VW model to choose from. If not supplied then the internally stored actions from the latest embed_actions() call will be used.
        """

        if "cb_label" in kwargs:
            chosen_action, cost, prob = kwargs["cb_label"]
        
        context_emb = self.context_embedding if "context" not in kwargs else kwargs["context"]
        action_embs = self.action_embeddings if "actions" not in kwargs else kwargs["actions"]

        if not context_emb or not action_embs:
            raise ValueError("Context and actions must be embedded first")

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
