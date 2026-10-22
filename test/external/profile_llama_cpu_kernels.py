#!/usr/bin/env python3
# Per kernel timing breakdown for the CPU llama-1B decode step (profiling helper, not part of the PR test).
import os
os.environ.setdefault("MAX_BUFFER_SIZE", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("DEBUG", "2")

from tinygrad import Tensor, Device
from extra.models.llama import Transformer

P = {"dim": 2048, "n_layers": 22, "n_heads": 32, "n_kv_heads": 4,
     "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 5632}

Tensor.manual_seed(0)
model = Transformer(max_context=1024, disable_kv_cache=True, jit=True, **P)
toks = Tensor([[1]])

def run(): model.forward_jit(toks, 0, 0.0, 0, 0.8, 0.0, 0.0).realize()

run()
print("=========== PROFILED RUN ===========", flush=True)
run()
print("=========== END ===========", flush=True)
