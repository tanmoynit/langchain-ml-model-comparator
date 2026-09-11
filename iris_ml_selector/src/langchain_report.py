"""
langchain_report.py
--------------------
This is the "LangChain" piece of the pipeline. Its job is to turn the raw
cross-validation results table into a short, readable recommendation of
which algorithm to use.

Two modes are supported:

1. LLM mode (real LangChain + an LLM)
   If an OPENAI_API_KEY environment variable is set (and langchain-openai
   is installed), we build a LangChain PromptTemplate -> ChatOpenAI chain
   and ask the model to write a short natural-language recommendation
   based on the results table.

2. Offline / deterministic mode (default, no API key needed)
   If no API key is configured, we still build the exact same LangChain
   PromptTemplate (so you can see/inspect the prompt that WOULD be sent),
   but instead of calling a paid LLM we render the report with a local
   rule-based formatter. This keeps the project fully "runnable out of
   the box" on any standalone computer with no internet/API key, while
   still demonstrating the LangChain prompt-construction pattern.

To switch to true LLM mode:
    1. pip install langchain-openai
    2. export OPENAI_API_KEY="sk-..."
    3. Re-run main.py -- it will automatically detect the key and use it.
"""

from __future__ import annotations
import os
import pandas as pd
from langchain_core.prompts import PromptTemplate

REPORT_PROMPT = PromptTemplate.from_template(
    """You are a helpful machine learning assistant. You are given the
results of a 5-fold cross validation comparison across several
classification algorithms on the "{dataset_name}" dataset.

Results table (higher is better for all metrics; std = standard deviation
across the 5 folds, lower std = more consistent performance):

{results_table}

Write a concise (5-8 sentence) recommendation for a data scientist:
1. Name the best-performing model and its mean accuracy.
2. Mention how consistent it was across folds (using the std values).
3. Briefly compare it to the second-best model.
4. Mention if any model is notably fast/slow (fit_time_mean_sec) which
   could matter for large datasets.
5. Give a one-line final recommendation.
"""
)


def _try_build_llm_chain():
    """
    Attempt to build a real LangChain LLM chain using ChatOpenAI.
    Returns the chain if an API key + package are available, else None.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        chain = REPORT_PROMPT | llm
        return chain
    except ImportError:
        print(
            "[langchain_report] OPENAI_API_KEY is set but langchain-openai "
            "is not installed. Falling back to offline report mode. "
            "Run: pip install langchain-openai"
        )
        return None


def _offline_report(results_df: pd.DataFrame, dataset_name: str) -> str:
    """
    Deterministic, rule-based version of the same report, used when no
    LLM/API key is configured. Mirrors the structure requested in
    REPORT_PROMPT so behavior is consistent either way.
    """
    best = results_df.iloc[0]
    second = results_df.iloc[1] if len(results_df) > 1 else None
    fastest = results_df.loc[results_df["fit_time_mean_sec"].idxmin()]
    slowest = results_df.loc[results_df["fit_time_mean_sec"].idxmax()]

    lines = []
    lines.append(
        f"Best model: **{best['model']}**, with a mean 5-fold accuracy of "
        f"{best['accuracy_mean']*100:.2f}% (± {best['accuracy_std']*100:.2f}%)."
    )
    if best["accuracy_std"] < 0.03:
        lines.append(
            "It was also very consistent across folds (low standard "
            "deviation), suggesting stable, reliable performance rather "
            "than a lucky split."
        )
    else:
        lines.append(
            "Its fold-to-fold variance was a bit higher than ideal, so "
            "results may shift somewhat depending on the data split."
        )

    if second is not None:
        diff = (best["accuracy_mean"] - second["accuracy_mean"]) * 100
        lines.append(
            f"The runner-up was **{second['model']}** at "
            f"{second['accuracy_mean']*100:.2f}% mean accuracy, "
            f"{diff:.2f} percentage points behind the top model."
        )

    if fastest["model"] != best["model"]:
        lines.append(
            f"Note that **{fastest['model']}** trained fastest "
            f"({fastest['fit_time_mean_sec']*1000:.2f} ms/fold on average), "
            "which could be preferable on much larger datasets even if its "
            "accuracy is slightly lower."
        )
    lines.append(
        f"Final recommendation: use **{best['model']}** for this dataset "
        "given it has the highest mean cross-validated accuracy; "
        "re-evaluate if your data size or distribution changes "
        "significantly."
    )
    return " ".join(lines)


def generate_recommendation(results_df: pd.DataFrame, dataset_name: str = "Iris") -> str:
    """
    Main entry point. Builds the LangChain prompt in all cases (so the
    prompt-construction pattern is always exercised), and either sends it
    to a real LLM or renders the offline fallback report.
    """
    table_str = results_df.to_string(index=False)

    chain = _try_build_llm_chain()
    if chain is not None:
        response = chain.invoke(
            {"dataset_name": dataset_name, "results_table": table_str}
        )
        # ChatOpenAI returns an AIMessage; extract text content.
        return getattr(response, "content", str(response))

    print(
        "[langchain_report] No OPENAI_API_KEY found -> generating the "
        "recommendation locally (offline mode). This still uses a "
        "LangChain PromptTemplate to structure the report; only the LLM "
        "call itself is swapped for a deterministic formatter."
    )
    return _offline_report(results_df, dataset_name)
