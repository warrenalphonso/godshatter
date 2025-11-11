import dataclasses

import numpy as np
from jaxtyping import Shaped
from tinygrad import nn
from tinygrad.dtype import dtypes
from tinygrad.nn.state import load_state_dict
from tinygrad.tensor import Tensor

from typechecker.tinygrad import Float, Int


@dataclasses.dataclass
class Config:
    vocab_size: int # V
    sequence_length: int # T
    n_layers: int # L
    n_heads: int # N
    d_head: int # H
    mlp_hidden_dim: int # F

    @property
    def d_model(self): # C
        return self.n_heads * self.d_head


TinyConfig = Config(
    vocab_size=50257,
    sequence_length=1024,
    n_layers=12,
    n_heads=12,
    d_head=64,
    mlp_hidden_dim= 3072
)

class Block:
    def __init__(self, config: Config):
        self.config = config
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = {
            "c_attn": nn.Linear(config.d_model, 3 * config.d_model), # 3 different linear projections (Q, K, V)
            "c_proj": nn.Linear(config.d_model, config.d_model),
        }
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = {
            "c_fc": nn.Linear(config.d_model, config.mlp_hidden_dim),
            "c_proj": nn.Linear(config.mlp_hidden_dim, config.d_model),
        }

    def __call__(self, x: Shaped[Tensor, "B T C"]) -> Shaped[Tensor, "B T C"]:
        B, T, C = x.shape
        qkv = self.attn["c_attn"](self.ln_1(x))
        assert isinstance(qkv, Shaped[Tensor, "B T 3*C"])
        q, k, v = qkv.split(C, dim=2)

        def to_heads(t: Shaped[Tensor, "B T C"]) -> Shaped[Tensor, "B N T (C//N)"]:
            return t.view(B, T, self.config.n_heads, self.config.d_head).transpose(1, 2).contiguous()

        q, k, v = map(to_heads, (q, k, v))
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.config.d_head))
        assert isinstance(att, Shaped[Tensor, "B N T T"])

        # Causal mask
        causal_mask = Tensor.ones((self.config.sequence_length, self.config.sequence_length), requires_grad=False).tril().view(1, 1, self.config.sequence_length, self.config.sequence_length)
        att = att.masked_fill(causal_mask[:, :, :T, :T] == 0, float('-inf')).softmax(axis=-1)
        assert isinstance(att, Shaped[Tensor, "B N T T"])
        y = att @ v

        # Merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.attn["c_proj"](y)
        # Add to residual
        x = x + y

        x = x + self.mlp["c_proj"](self.mlp["c_fc"](self.ln_2(x)).gelu())
        return x

class GPT2:
    def __init__(self, config: Config):
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.sequence_length, config.d_model)
        self.h = [Block(config) for _ in range(config.n_layers)]
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    @classmethod
    def load_from_huggingface(cls):
        from transformers import GPT2Config, GPT2Model

        model = cls(TinyConfig)
        model_hf = GPT2Model.from_pretrained("gpt2") # 124M

        sd = {}
        for k, v in model_hf.state_dict().items():
            if k.startswith("transformer."): k = k.split("transformer.")[1]
            # Huggingface dimensions for linear layers are transposed
            if any(x in k for x in ["c_attn.weight", "c_proj.weight", "c_fc.weight", "c_proj.weight"]):
                v = v.t()
            sd[k] = Tensor(v.cpu().numpy())
        sd['lm_head.weight'] = sd['wte.weight'] # Weight tying
        load_state_dict(model, sd)
        return model

    def __call__(self, tokens: Int[Tensor, "B T"]) -> Float[Tensor, "B T V"]:
        B, T = tokens.shape
        assert T <= self.config.sequence_length, "Input sequence length exceeds limit"

        x = self.wte(tokens)
        pos = Tensor.arange(T, dtype=dtypes.long)
        x = x + self.wpe(pos)

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x
