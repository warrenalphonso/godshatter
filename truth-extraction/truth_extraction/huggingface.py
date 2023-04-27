"""Huggingface transformers and their tokenizers."""
from typing import TYPE_CHECKING

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

if TYPE_CHECKING:
    from .model import Model, ModelType

# Shortcuts for 🤗 models
HUGGINGFACE_MODELS = {
    "gpt-j": "EleutherAI/gpt-j-6B",
    "T0pp": "bigscience/T0pp",
    "unifiedqa": "allenai/unifiedqa-t5-11b",
    "T5": "t5-11b",
    "deberta-mnli": "microsoft/deberta-xxlarge-v2-mnli",
    "deberta": "microsoft/deberta-xxlarge-v2",
    "roberta-mnli": "roberta-large-mnli",
}


def load_model(model_name: str, parallelize: bool = False, device: str | None = "cuda") -> "Model":
    """
    Load HuggingFace model.

    Args:
        parallelize: distribute model across all available devices
            https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/t5#transformers.T5Model.parallelize
        device: move model to device
    """
    from .model import Model

    if model_name in HUGGINGFACE_MODELS:
        model_name = HUGGINGFACE_MODELS[model_name]

    model_type: "ModelType"
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model_type = "encoder_decoder"
    except:
        try:
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            model_type = "encoder"
        except:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model_type = "decoder"

    # Set model to evaluation mode
    model.eval()

    if parallelize:
        model.parallelize()
    elif device:
        model.to(device)

    # Set maximum number of tokens for input to 512 so that we can pad
    # TODO: we only get a log message if input tokens exceed this. Do we want it
    # to fail loudly?
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

    return Model(transformer=model, tokenizer=tokenizer, model_type=model_type, device=device)
