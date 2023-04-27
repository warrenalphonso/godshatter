"""Load an LLM."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict, TypeVar

import torch

ModelType = Literal["encoder_decoder", "decoder", "encoder"]


class TransformerOutput(TypedDict):
    hidden_state: Any
    output: Any


@dataclass
class Model:
    tokenizer: Callable[[str], Any]
    transformer: Callable[[Any], TransformerOutput]
    model_type: ModelType
    device: str | None


def extract_hidden_states(model: Model, prompt: str, layer: int = -1, token_index: int = -1):
    """
    You have model and some tokenized inputs. Now return hidden states for a specific
    layer or all layers.

    If "specify_encoder" is set, use "encoder_hidden_states" instead of "hidden_states"
    - so model type is relevant
    This isn't necessary for encoder-only or decoder-only models.

    Encoder-only (aka auto-encoding model): attention can access all words. Used
        for tasks that require an understanding of full sentence
    Decoder-only (aka autoregressive models): attention can only access words before
        it in sentence. Used for text generation.
    encoder-decoder (aka sequence-to-sequence): encoder layers can see everything,
        but decoder can only see words before. Used for generating new sentences
        which depend on understanding something. (Why isn't this same as decoder-only?)

    First let's see if I can get to understanding what the full hidden state tensor
    for a single layer means. The final conditional is just getting some subset of
    this, apparently subset relevant for one token.
    """

    with torch.no_grad():
        batch_ids = model.tokenizer(
            prompt
        )  # , truncation=True, padding="max_length", return_tensors="pt")
        # if model.device is not None:
        #     batch_ids = batch_ids.to(model.device)
        output = model.transformer(batch_ids)  #  , output_hidden_states=True)

    hs_ = output["hidden_state"]  # Full hidden state
    _hss_1 = hs_.shape
    print(
        f"Full hidden state dimension is {hs_.shape} (layers, batch size, sequence "
        f"length, dimension)"
    )

    # Get hidden state for a single layer
    # TODO: Allow getting hidden state for all layers
    hs_ = hs_[layer]  # (bs, seq_len, dim)
    _hss_2 = hs_.shape
    assert len(_hss_2) == 3
    assert _hss_1[1:] == _hss_2

    hs_ = hs_.unsqueeze(-1)  # (bs, seq_len, dim, 1)
    _hss_3 = hs_.shape
    assert len(_hss_3) == 4
    assert _hss_3[-1] == 1
    assert _hss_3[:-1] == _hss_2

    hs_ = hs_.detach().cpu()

    # Get hidden state for a single token
    # Get part that isn't padding, then get index of token_index of that. That is,
    # if token_index==-1, it should get last index that isn't padding!
    token_index_ = 0
    hs_ = hs_[torch.arange(hs_.size(0)), token_index_]  # (bs, dim, 1)
    _hss_4 = hs_.shape
    assert _hss_3[0] == _hss_4[0]
    assert _hss_3[2] == _hss_4[1]
    assert _hss_4[2] == 1
    assert len(_hss_4) == 3

    return hs_
