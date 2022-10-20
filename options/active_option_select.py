from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
from baal.active import heuristics

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
}

HEURISTIC_METHODS = {
    "bald": heuristics.BALD,
    "random": heuristics.Random,
    "entropy": heuristics.Entropy,
    "variance": heuristics.Variance,
    "margin": heuristics.Margin,
    "certainty": heuristics.Certainty,
}