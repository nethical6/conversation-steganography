#!/usr/bin/env python3
"""Persistent deterministic Hugging Face backend for decalgo.

Install: python3 -m pip install torch transformers
Run indirectly through: decalgo generate ... / decalgo extract ...
"""

import argparse
import hashlib
import json
import sys


def reply(**values):
    print(json.dumps({"ok": True, **values}, separators=(",", ":")), flush=True)


def fail(exc):
    print(json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")), flush=True)


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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype)
    model.to(args.device)
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None) or args.revision
    identity = json.dumps({
        "model": args.model,
        "revision": resolved_revision,
        "tokenizer": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }, sort_keys=True).encode()
    fingerprint = "hf:" + hashlib.sha256(identity).hexdigest()
    special = set(tokenizer.all_special_ids)
    for token, token_id in tokenizer.get_vocab().items():
        if token.startswith("<|") and token.endswith("|>"):
            special.add(token_id)
    token_text_cache = {}

    def token_text(token_id):
        if token_id not in token_text_cache:
            token_text_cache[token_id] = tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
        return token_text_cache[token_id]

    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request["op"]
            if op == "info":
                reply(fingerprint=fingerprint)
            elif op == "tokenize":
                ids = tokenizer.encode(request.get("text", ""), add_special_tokens=False)
                reply(tokens=ids)
            elif op == "detokenize":
                text = tokenizer.decode(request.get("tokens", []), skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                reply(text=text)
            elif op == "next":
                ids = request["tokens"]
                top_n = int(request["top_n"])
                visible_tokens = request.get("visible_tokens")
                if not ids:
                    raise ValueError("model context cannot be empty")
                input_ids = torch.tensor([ids], device=args.device)
                with torch.inference_mode():
                    logits = model(input_ids=input_ids).logits[0, -1].float()
                if special:
                    logits[list(special)] = -torch.inf
                pool_n = top_n if visible_tokens is None else min(logits.shape[-1], top_n * 4)
                values, indices = torch.topk(logits, k=pool_n, sorted=True)
                candidates = []
                tail = list(visible_tokens[-8:]) if visible_tokens else []
                for i, v in zip(indices.tolist(), values.tolist()):
                    i = int(i)
                    if visible_tokens is not None:
                        probe = tail + [i]
                        probe_text = tokenizer.decode(probe, skip_special_tokens=False,
                                                      clean_up_tokenization_spaces=False)
                        encoded = tokenizer.encode(probe_text, add_special_tokens=False)
                        if encoded != probe:
                            continue
                    candidates.append({"id": i, "score": float(v), "text": token_text(i)})
                    if len(candidates) == top_n:
                        break
                if len(candidates) != top_n:
                    raise ValueError(f"only {len(candidates)} copy-safe candidates available; need {top_n}")
                reply(candidates=candidates)
            else:
                raise ValueError("unknown operation: " + str(op))
        except Exception as exc:
            fail(exc)


if __name__ == "__main__":
    main()
