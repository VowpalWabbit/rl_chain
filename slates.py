import vowpal_wabbit_next as vw
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict
from itertools import chain
import random


#TODO: remove copy-paste
def parse_lines(parser: vw.TextFormatParser, input_str: str) -> List[vw.Example]:
    return [parser.parse_line(line) for line in input_str.split("\n")]


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
    label: Optional[Label]

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


class Policy(ABC):
    @abstractmethod
    def predict(self, decision: Decision) -> Label:
        ...


class VwPolicy:
    def __init__(self, workspace: vw.Workspace, *_, **__):
        self.workspace = workspace
    
    def predict(self, decision: Decision) -> Label:
        text_parser = vw.TextFormatParser(self.workspace)
        return Label(self.workspace.predict_one(parse_lines(text_parser, decision.vwtxt)))


class RandomPolicy:
    def __init__(self, *_, **__):
        ...

    def predict(self, decision: Decision) -> Label:
        return Label([[(random.randint(0, len(slot) - 1), 1.0 / len(slot))] for slot in decision.actions])


class FirstChoicePolicy:
    def __init__(self, *_, **__):
        ...

    def predict(self, decision: Decision) -> Label:
        return Label([[(0, 1)] for slot in decision.actions])