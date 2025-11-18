import dataclasses
import pathlib
from datetime import datetime

import numpy as np
from jaxtyping import Shaped
from tinygrad import TinyJit, nn
from tinygrad.dtype import dtypes
from tinygrad.helpers import Timing, fetch
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save
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
        logits = self.lm_head(x)
        return logits

class DataLoader:
    def __init__(self, B, T):
        import tiktoken

        self.B = B
        self.T = T

        with open(fetch("https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt")) as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        self.tokens = Tensor(enc.encode(text), dtype='int')
        self.pos = 0

    def next_batch(self) -> tuple[Int[Tensor, "B T"], Int[Tensor, "B T"]]:
        needed = self.B*self.T+1
        if self.pos + needed > len(self.tokens):
            self.pos = 0

        tokens = self.tokens[self.pos : self.pos+needed]
        x = tokens[:-1].view((self.B, self.T)).contiguous()
        y = tokens[1:].view((self.B, self.T)).contiguous()

        self.pos += self.B*self.T
        return x.realize(), y.realize()


if __name__ == "__main__":
    model = GPT2(TinyConfig)
    dl = DataLoader(8, 1024)
    opt = AdamW(get_parameters(model), lr=3e-4)
    checkpoints_dir = pathlib.Path("/mydata/nanogpt-test-ckpts")
    checkpoints_dir.mkdir(exist_ok=True)

    @TinyJit
    def step(tokens: Int[Tensor, "B T"], targets: Int[Tensor, "B T"]):
        logits = model(tokens)
        loss = logits.sparse_categorical_crossentropy(targets)
        opt.zero_grad()
        loss.backward()
        # schedule updates into same graph instead of updating state in jitted function
        return loss.realize(*opt.schedule_step())

    with Tensor.train():
        for i in range(1, 50_001):
            with Timing(f"Time (step {i}): "):
                x, y = dl.next_batch()
                loss = step(x.contiguous(), y.contiguous()).item()
                print(f'{loss=}')

            if i % 1_000 == 0:
                ts = datetime.utcnow().strftime("%y%m%d-%H%M")
                safe_save(get_state_dict(model), str(checkpoints_dir / f"step_{ts}_{i}.safetensors"))
