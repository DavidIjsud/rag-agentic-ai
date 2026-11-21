"""Week 0 build: how thinking level and temperature change a Gemini answer."""

import time
import warnings
from importlib import import_module

from langchain_core.exceptions import ModelError, ModelRateLimitError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tracers.langchain import wait_for_all_tracers

from rag_agentic_ai.costs import PRICES, cost_usd  # pyright: ignore[reportMissingImports]
from rag_agentic_ai.llm import get_model  # pyright: ignore[reportMissingImports]

MESSAGES = [
    SystemMessage("Eres el asistente de una zapatería de La Paz. Responde en dos frases."),
    HumanMessage("¿Qué zapatillas me recomiendas para caminar todo el día?"),
]
FLASH = "google_genai:gemini-3.5-flash"
LITE = "google_genai:gemini-3.5-flash-lite"

# (label, model, settings): 9 calls in all
RUNS = [
    *[(f"A thinking={level}", FLASH, {"thinking_level": level})
      for level in ("minimal", "low", "medium", "high")],
    ("B temperature=0 #1", FLASH, {"temperature": 0}),
    ("B temperature=0 #2", FLASH, {"temperature": 0}),
    ("B default #1", FLASH, {}),
    ("B default #2", FLASH, {}),
    ("C lite temperature=0", LITE, {"temperature": 0}),
]


def run_one(label: str, spec: str, settings: dict) -> float:
    model = get_model(spec, timeout=90, max_retries=2, **settings)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")  # record every warning, even repeats
        reply = model.invoke(MESSAGES)
    seconds = time.perf_counter() - started

    usage = reply.usage_metadata or {}
    tokens_in = usage.get("input_tokens", 0)
    tokens_out = usage.get("output_tokens", 0)  # already includes reasoning
    reasoning = usage.get("output_token_details", {}).get("reasoning", 0)
    price = PRICES.get(spec.partition(":")[2])
    cost = cost_usd(tokens_in, tokens_out, *price) if price else 0.0

    answer = reply.text.replace("\n", " ")[:60]
    print(f"{label:<22} {seconds:5.1f}s {tokens_in:5} {tokens_out:5} "
          f"{reasoning:5}  ${cost:.6f}  {answer}")
    for warning in caught:
        print(f"{'':<22} ⚠️  {warning.message}")
    return cost


def main() -> None:
    print(f"{'setting':<22} {'time':>6} {'in':>5} {'out':>5} {'think':>5}  {'cost':<9}  answer")
    total = 0.0
    for label, spec, settings in RUNS:
        try:
            total += run_one(label, spec, settings)
        except ModelRateLimitError:
            print("\nFree-tier limit reached (429 RESOURCE_EXHAUSTED). "
                  "Wait a minute and run it again, or wait until tomorrow if it's the daily limit.")
            break
        except ModelError as error:
            print(f"{label:<22} failed: {type(error).__name__}: {str(error)[:80]}")
    print(f"\nTotal cost on a paid plan: ${total:.6f}")
    wait_for_all_tracers()  # send the traces to LangSmith before the script exits


if __name__ == "__main__":
    main()