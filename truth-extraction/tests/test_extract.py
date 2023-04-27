import pytest
import torch
from torch import nn
from truth_extraction.ccs import CCS
from truth_extraction.model import Model, extract_hidden_states

# TODO: Extract hidden states for each of these transformers. I think I need to
# use PyTorch forward hooks.


@pytest.fixture
def encoder_model():
    def tokenizer(string):
        return torch.rand(20, 32, 512)

    def transformer(encoding):
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        activations = []

        def hook(model, input, output):
            activations.append(output.detach())

        for _name, module in transformer_encoder.layers.named_children():
            module.register_forward_hook(hook)
        output = transformer_encoder(encoding)
        return {"output": output, "hidden_state": torch.stack(activations)}

    return Model(tokenizer=tokenizer, transformer=transformer, model_type="encoder", device=None)


@pytest.fixture
def decoder_model():
    def tokenizer(string):
        return torch.rand(20, 32, 512)

    def transformer(encoding):
        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        # Sequence from last layer of encoder
        memory = torch.zeros(10, 32, 512)
        activations = []

        def hook(model, input, output):
            activations.append(output.detach())

        for _name, module in transformer_decoder.layers.named_children():
            module.register_forward_hook(hook)
        output = transformer_decoder(encoding, memory)
        return {"output": output, "hidden_state": torch.stack(activations)}

    return Model(tokenizer=tokenizer, transformer=transformer, model_type="decoder", device=None)


@pytest.fixture
def encoder_decoder_model():
    def tokenizer(string):
        return {
            "question_tokens": torch.rand(10, 32, 512),
            "answer_tokens": torch.rand(20, 32, 512),
        }

    def transformer(input_):
        transformer_ = nn.Transformer()
        activations = []

        def hook(model, input, output):
            activations.append(output.detach())

        # Get hidden states for decoder -- have to choose one of encoder or decoder
        # since dimensions are different
        for _name, module in transformer_.decoder.layers.named_children():
            module.register_forward_hook(hook)

        output = transformer_(input_["question_tokens"], input_["answer_tokens"])
        return {"output": output, "hidden_state": torch.stack(activations)}

    return Model(
        tokenizer=tokenizer, transformer=transformer, model_type="encoder_decoder", device=None
    )


@pytest.fixture
def models(encoder_model, decoder_model, encoder_decoder_model):
    return [encoder_model, decoder_model, encoder_decoder_model]


def test_extract(models):
    for model in models:
        extract_hidden_states(model=model, prompt="hello!")


def test_css():
    pass
