# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Show how an external grounding layer can hand off only generation-required work.

The external system owns retrieval, evidence policy, and the decision to either
complete the request deterministically or ask the existing ORT GenAI runtime to
generate from already-approved messages. ORT GenAI continues to own tokenization,
chat templating, model execution, search/sampling, and KV state.

Example delivery JSON:
  {"action":"complete","reply":{"a":"17 cm","disposition":"answer"}}

or:
  {"action":"generate","messages":[{"role":"system","content":"..."},
                                     {"role":"user","content":"..."}]}
"""

import argparse
import json
from pathlib import Path

import onnxruntime_genai as og


def load_delivery(path: Path) -> dict:
    delivery = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(delivery, dict):
        raise ValueError("delivery must be a JSON object")
    action = delivery.get("action")
    if action not in {"complete", "generate"}:
        raise ValueError("delivery.action must be 'complete' or 'generate'")
    return delivery


def run_delivery(delivery: dict, model_path: str, max_new_tokens: int) -> None:
    if delivery["action"] == "complete":
        reply = delivery.get("reply")
        if not isinstance(reply, dict):
            raise ValueError("complete delivery requires a reply object")
        print(json.dumps(reply, ensure_ascii=False))
        return

    messages = delivery.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("generate delivery requires non-empty messages")

    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    prompt = tokenizer.apply_chat_template(
        messages=json.dumps(messages, ensure_ascii=False),
        add_generation_prompt=True,
    )
    input_tokens = tokenizer.encode(prompt)

    params = og.GeneratorParams(model)
    params.set_search_options(
        max_length=len(input_tokens) + max_new_tokens,
        batch_size=1,
        do_sample=False,
    )
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    input_length = len(generator.get_sequence(0))

    while not generator.is_done():
        generator.generate_next_token()

    sequence = generator.get_sequence(0)
    print(tokenizer.decode(sequence[input_length:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max_new_tokens must be positive")
    run_delivery(load_delivery(args.delivery), args.model_path, args.max_new_tokens)


if __name__ == "__main__":
    main()
