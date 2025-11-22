import dataclasses
import math
import time

import torch
import urllib3
from jaxtyping import Float, Int
from torch import Tensor, nn
from torch.nn import functional as F


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

SmallConfig = Config(
    vocab_size=51200,
    sequence_length=1024,
    n_layers=12,
    n_heads=12,
    d_head=64,
    mlp_hidden_dim= 3072
)

class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, bias=True, scale=None):
        super().__init__(in_features, out_features, bias=bias)
        std = 0.02
        if scale: std *= scale
        torch.nn.init.normal_(self.weight, std=std)
        if bias:
            torch.nn.init.zeros_(self.bias)

class Embedding(nn.Embedding):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__(vocab_size, embed_size)
        torch.nn.init.normal_(self.weight, std=0.02)

class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        residual_scaling_factor = 1.0 / math.sqrt(2 * config.n_layers)
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = nn.ModuleDict({
            "c_attn": Linear(config.d_model, 3 * config.d_model), # 3 different linear projections (Q, K, V)
            "c_proj": Linear(config.d_model, config.d_model, scale=residual_scaling_factor),
        })
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.ModuleDict({
            "c_fc": Linear(config.d_model, config.mlp_hidden_dim),
            "c_proj": Linear(config.mlp_hidden_dim, config.d_model, scale=residual_scaling_factor),
        })

        self.register_buffer("causal_mask", torch.tril(torch.ones((config.sequence_length, config.sequence_length))).view(1, 1, config.sequence_length, config.sequence_length))

    def forward(self, x: Float[Tensor, "B T C"]) -> Float[Tensor, "B T C"]:
        B, T, C = x.shape
        qkv = self.attn["c_attn"](self.ln_1(x))
        q, k, v = qkv.split(C, dim=2)

        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        x = x + self.attn["c_proj"](y)

        x = x + self.mlp["c_proj"](F.gelu(self.mlp["c_fc"](self.ln_2(x))))

        return x

class GPT2(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.wte = Embedding(config.vocab_size, config.d_model)
        self.wpe = Embedding(config.sequence_length, config.d_model)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, tokens: Int[Tensor, "B T"]) -> Float[Tensor, "B T V"]:
        B, T = tokens.shape
        x = self.wte(tokens) + self.wpe(torch.arange(0, T, dtype=torch.long, device=tokens.device))

        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

def dataloader(B, T):
    import tiktoken

    text = urllib3.PoolManager().request("GET", "https://raw.githubusercontent.com/karpathy/char-rnn/refs/heads/master/data/tinyshakespeare/input.txt").data.decode()
    enc = tiktoken.get_encoding("gpt2")
    alltokens = torch.as_tensor(enc.encode(text), dtype=torch.long)
    pos = 0
    needed = B*T + 1

    while pos + needed < len(alltokens):
        tokens = alltokens[pos : pos+needed]
        x = tokens[:-1].view((B, T))
        y = tokens[1:].view((B, T))

        pos += B*T
        yield x, y

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using {device=}")
    torch.set_default_device(device)
    torch.set_float32_matmul_precision("high")

    model = GPT2(SmallConfig)
    model.to(device)
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for epoch in range(10):
        print(f"Epoch {epoch+1}/10")
        for i, (x, y) in enumerate(dataloader(16, 1024)):
            t0 = time.time()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize() if device == "cuda" else None

            t1 = time.time()
            print(f"({(t1-t0)*1000:.1f}ms) {i=}, {loss.item()=}")
