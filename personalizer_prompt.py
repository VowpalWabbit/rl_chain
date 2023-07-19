from langchain.prompts.prompt import PromptTemplate

_PROMPT_TEMPLATE = """You are given a set of action(s), here are their description(s): {selected}.

You have to embed these action(s) into the text where it makes sense. Here is the text: {text_to_personalize}.

Don't leave any action(s) out.
"""


PROMPT = PromptTemplate(
    input_variables=["selected", "text_to_personalize"],
    template=_PROMPT_TEMPLATE,
)
