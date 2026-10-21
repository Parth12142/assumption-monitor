
"""
Use LFM2-1.2B to automatically review assumption candidates.

Input:
    data/labels/candidates.csv

Output:
    data/labels/llm_labels.csv

The model follows docs/labeling_guide.md and returns:
    is_assumption = 1 or 0
    topic = one of the allowed topics or ""
    review_status = single / needs_review / reviewed
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "LiquidAI/LFM2-1.2B"

TOPICS = [
    "fuel",
    "supplier",
    "demand",
    "rates",
    "currency",
    "labor",
    "regulation",
    "other",
]

SYSTEM_PROMPT = """
You are labeling sentences from company annual reports.

Your task is to decide whether ONE sentence is an ASSUMPTION.

Use these rules exactly.

A sentence is an ASSUMPTION only when ALL THREE are true:

1. It looks forward. It says what will happen, what is expected,
   or what the plan takes for granted.

2. It is about something outside the company's own control:
   prices, rates, demand, suppliers, laws, wages, exchange rates,
   inflation, weather, technology, competitors, etc.

3. The plan depends on it, or it could be checked later against data.

Typical assumption signals:
- we assume
- we expect ... to remain
- is projected to
- based on the assumption that
- we anticipate prices will

A sentence is NOT an assumption when it is:

- History
- A description
- Legal boilerplate
- A risk or possibility with no stated expectation
- A plan or promise about the company's own actions
- An expectation about the company's own results with no outside driver
- Only a topic mention
- An opinion about the present

Examples:

"We expect interest rates to remain low, which supports our borrowing plan."
=> assumption, topic rates

"Interest rates may rise."
=> not an assumption

"We expect to complete the acquisition in the third quarter."
=> not an assumption

"Our budget assumes inflation of 3%."
=> assumption, topic other

"We believe our facilities are adequate."
=> not an assumption

Topics:

fuel = oil, gas, diesel, electricity and energy prices
supplier = suppliers, raw material supply, delivery, supply chain
demand = customers, sales volume, market growth, consumer spending
rates = interest rates and borrowing costs
currency = exchange rates
labor = wages, hiring, workforce, labor supply
regulation = laws, taxes, tariffs, rules and approvals
other = inflation, weather, technology, competitors, or anything else

If there are multiple topics, select the FIRST main topic mentioned.

Return ONLY valid JSON.

Required JSON format:

{
  "is_assumption": 0,
  "topic": "",
  "review_status": "single",
  "reason": "short explanation"
}

For an assumption:

{
  "is_assumption": 1,
  "topic": "rates",
  "review_status": "single",
  "reason": "short explanation"
}

If you genuinely cannot decide, use:

{
  "is_assumption": "",
  "topic": "",
  "review_status": "needs_review",
  "reason": "short explanation"
}

Do not add markdown.
Do not add text before or after the JSON.
"""


def load_model():
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    model.eval()

    print("Model loaded.")
    return tokenizer, model


def make_prompt(sentence: str, section: str) -> str:
    return f"""
{SYSTEM_PROMPT}

Section:
{section}

Sentence:
{sentence}

Return ONLY the JSON object.
"""


def extract_json(text: str):
    """
    Try to extract the first JSON object from model output.
    """
    text = text.strip()

    # Remove markdown fences if the model accidentally adds them.
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_result(result):
    """
    Make sure the model output matches our label format.
    """
    if not isinstance(result, dict):
        return {
            "is_assumption": "",
            "topic": "",
            "review_status": "needs_review",
            "reason": "Model did not return valid JSON.",
        }

    value = str(result.get("is_assumption", "")).strip()

    if value in {"0", "1"}:
        is_assumption = value
    else:
        is_assumption = ""

    topic = str(result.get("topic", "")).strip().lower()

    if is_assumption == "1":
        if topic not in TOPICS:
            return {
                "is_assumption": "",
                "topic": "",
                "review_status": "needs_review",
                "reason": "Model selected an invalid topic.",
            }
    else:
        topic = ""

    status = str(
        result.get("review_status", "single")
    ).strip().lower()

    if status not in {"single", "needs_review", "reviewed"}:
        status = "needs_review"

    if is_assumption == "":
        status = "needs_review"

    reason = str(result.get("reason", "")).strip()

    return {
        "is_assumption": is_assumption,
        "topic": topic,
        "review_status": status,
        "reason": reason,
    }


def classify(tokenizer, model, sentence, section):
    prompt = make_prompt(sentence, section)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    result = extract_json(text)

    if result is None:
        return {
            "is_assumption": "",
            "topic": "",
            "review_status": "needs_review",
            "reason": f"Could not parse model output: {text[:300]}",
        }

    return normalize_result(result)


def run(
    candidates_path,
    output_path,
    start=0,
    limit=None,
):
    candidates = pd.read_csv(
        candidates_path,
        dtype=str,
    ).fillna("")

    required = [
        "sentence_id",
        "doc_id",
        "section",
        "sentence",
    ]

    missing = [
        column
        for column in required
        if column not in candidates.columns
    ]

    if missing:
        raise ValueError(
            f"Candidates file is missing columns: {missing}"
        )

    tokenizer, model = load_model()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Resume automatically if output already exists.
    if output_path.exists():
        results = pd.read_csv(
            output_path,
            dtype=str,
        ).fillna("")

        done = set(results["sentence_id"])

        print(
            f"{len(done)} sentences already reviewed."
        )
    else:
        results = pd.DataFrame(
            columns=[
                "sentence_id",
                "doc_id",
                "section",
                "sentence",
                "is_assumption",
                "topic",
                "labeler",
                "label_version",
                "review_status",
                "is_synthetic",
                "reason",
            ]
        )

        done = set()

    rows = candidates.iloc[start:]

    if limit is not None:
        rows = rows.head(limit)

    total = len(rows)

    for number, (_, row) in enumerate(
        rows.iterrows(),
        start=1,
    ):
        sentence_id = row["sentence_id"]

        if sentence_id in done:
            continue

        print()
        print("=" * 80)
        print(f"{number}/{total}")
        print(sentence_id)
        print(row["sentence"])
        print("-" * 80)

        result = classify(
            tokenizer,
            model,
            row["sentence"],
            row["section"],
        )

        new_row = {
            "sentence_id": sentence_id,
            "doc_id": row["doc_id"],
            "section": row["section"],
            "sentence": row["sentence"],
            "is_assumption": result["is_assumption"],
            "topic": result["topic"],
            "labeler": "lfm2-1.2b",
            "label_version": "llm-v1",
            "review_status": result["review_status"],
            "is_synthetic": "False",
            "reason": result["reason"],
        }

        results = pd.concat(
            [
                results,
                pd.DataFrame([new_row]),
            ],
            ignore_index=True,
        )

        # Save after EVERY sentence.
        results.to_csv(
            output_path,
            index=False,
        )

        print(
            f"RESULT: assumption={result['is_assumption']} "
            f"topic={result['topic']} "
            f"status={result['review_status']}"
        )
        print(f"Reason: {result['reason']}")

    print()
    print(
        f"Finished. Saved {len(results)} reviews to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidates",
        default="data/labels/candidates.csv",
    )

    parser.add_argument(
        "--output",
        default="data/labels/llm_labels.csv",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    run(
        args.candidates,
        args.output,
        args.start,
        args.limit,
    )


if __name__ == "__main__":
    main()
