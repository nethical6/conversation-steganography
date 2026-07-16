#!/usr/bin/env python3
"""Persistent deterministic MLX-LM backend for decalgo on Apple Silicon."""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path


def reply(**values):
    print(json.dumps({"ok": True, **values}, separators=(",", ":")), flush=True)


def fail(exc):
    print(json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")), flush=True)


def weight_identity(model_name):
    path = Path(model_name).expanduser()
    if not path.is_dir():
        return model_name
    files = []
    for item in sorted(path.glob("model*.safetensors")):
        resolved = item.resolve()
        files.append(resolved.name + ":" + str(resolved.stat().st_size))
    return files


DISALLOWED_WORDS = {
    "assistant", "example", "format", "input", "instruction", "instructions",
    "message", "messages", "metadata", "note", "output", "prompt", "prompts",
    "recipient", "recipients", "response", "role", "sender", "sent", "system",
    "timestamp", "transcript", "user", "analysis",
}


def ordinary_visible_token(text):
    if not text or any(ch in text for ch in '\r\n\t0123456789#{}[]()<>*`|\\\":~&=+_^%$@/“”„‟«»（）【】'):
        return False
    return text.lower().strip(" .,!?;'-_") not in DISALLOWED_WORDS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    import mlx.core as mx
    import mlx_lm
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(args.model, revision=args.revision)
    tokenizer_impl = getattr(tokenizer, "_tokenizer", tokenizer)
    identity = json.dumps({
        "runtime": "mlx",
        "model": weight_identity(args.model),
        "revision": args.revision,
        "tokenizer": tokenizer_impl.__class__.__name__,
        "vocab_size": len(tokenizer_impl),
        "mlx": getattr(mx, "__version__", "unknown"),
        "mlx_lm": getattr(mlx_lm, "__version__", "unknown"),
    }, sort_keys=True).encode()
    fingerprint = "mlx:" + hashlib.sha256(identity).hexdigest()
    special = set(getattr(tokenizer_impl, "all_special_ids", []))
    token_text_cache = {}

    def token_text(token_id):
        if token_id not in token_text_cache:
            token_text_cache[token_id] = tokenizer_impl.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        return token_text_cache[token_id]

    for token, token_id in tokenizer_impl.get_vocab().items():
        if token.startswith("<|") and token.endswith("|>"):
            special.add(token_id)
    cached_tokens = None
    prompt_cache = None
    # Snapshot of the KV state at the start of the most recent generation
    # (after processing the prompt prefix, before any generated tokens).
    # Reused to skip re-prefilling the identical prompt for every trial.
    gen_start_tokens = None
    gen_start_cache = None

    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "info":
                reply(fingerprint=fingerprint)
            elif op == "tokenize":
                ids = tokenizer_impl.encode(request.get("text", ""), add_special_tokens=False)
                reply(tokens=ids)
            elif op == "detokenize":
                text = tokenizer_impl.decode(request.get("tokens", []), skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)
                reply(text=text)
            elif op == "next":
                ids = request["tokens"]
                top_n = int(request["top_n"])
                visible_tokens = request.get("visible_tokens")
                if not ids:
                    raise ValueError("model context cannot be empty")

                if cached_tokens is not None and ids[:-1] == cached_tokens:
                    # Continue current generation — only decode the new token.
                    logits = model(mx.array([[ids[-1]]]), cache=prompt_cache)[0, -1]
                elif gen_start_tokens is not None and ids[:-1] == gen_start_tokens:
                    # New trial with the same prompt prefix — restore the KV
                    # snapshot so we avoid re-prefilling the whole context.
                    prompt_cache = copy.deepcopy(gen_start_cache)
                    logits = model(mx.array([[ids[-1]]]), cache=prompt_cache)[0, -1]
                else:
                    # New context: full prefill required.
                    prompt_cache = make_prompt_cache(model)
                    gen_start_tokens = None
                    gen_start_cache = None
                    if len(ids) > 1:
                        model(mx.array([ids[:-1]]), cache=prompt_cache)
                        mx.eval()  # materialise cache before snapshotting
                        gen_start_tokens = list(ids[:-1])
                        gen_start_cache = copy.deepcopy(prompt_cache)
                    logits = model(mx.array([[ids[-1]]]), cache=prompt_cache)[0, -1]
                mx.eval(logits)
                scores = np.asarray(logits, dtype=np.float32)
                if special:
                    scores[list(special)] = -np.inf
                pool_n = top_n if visible_tokens is None else min(len(scores), top_n * 4)
                indices = np.argpartition(-scores, pool_n - 1)[:pool_n]
                indices = sorted(indices, key=lambda i: (-float(scores[i]), int(i)))
                candidates = []
                tail = list(visible_tokens[-8:]) if visible_tokens else []
                for i in indices:
                    i = int(i)
                    if visible_tokens is not None:
                        probe = tail + [i]
                        probe_text = tokenizer_impl.decode(probe, skip_special_tokens=False,
                                                           clean_up_tokenization_spaces=False)
                        encoded = tokenizer_impl.encode(probe_text, add_special_tokens=False)
                        if encoded != probe:
                            continue
                    candidates.append({"id": i, "score": float(scores[i]), "text": token_text(i)})
                    if len(candidates) == top_n:
                        break
                if len(candidates) != top_n:
                    raise ValueError(f"only {len(candidates)} copy-safe candidates available; need {top_n}")
                cached_tokens = list(ids)
                reply(candidates=candidates)
            else:
                raise ValueError("unknown operation: " + str(op))
        except Exception as exc:
            fail(exc)


if __name__ == "__main__":
    main()
