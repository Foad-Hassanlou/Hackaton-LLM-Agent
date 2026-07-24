"""Tools registered on the agents.

`perform_search` and `handle_results` are built by `build_agent_tools` so they
can close over the searcher, the shared context and the agents they hand off
to. Their names, signatures and docstrings are part of the tool schema AutoGen
sends to the model — do not rename or reword them casually.
"""

import json
import re
from typing import Callable, List, Tuple

from autogen import ConversableAgent
from autogen.agentchat.group import AgentTarget, ContextVariables, ReplyResult

from hackaton.search import KeywordSearcher


def search_products(searcher: KeywordSearcher, query: str, k: int = 5) -> List[dict]:
    """Run the pure-Python keyword search and normalise it to dicts.

    Returns a list of dicts with fields: category, row_idx, score, and doc.
    """
    raw_results = searcher.normal_search(query, k=k)
    return [
        {
            "category": category,
            "row_idx": row_idx,
            "score": score,
            "doc": doc
        }
        for score, category, row_idx, doc in raw_results
    ]


def pro_search_products(searcher: KeywordSearcher, query: str, k: int = 5) -> List[dict]:
    """Run the ChromaDB keyword search and normalise it to dicts.

    Returns a list of dicts with fields: category, row_idx, score, and doc.
    """
    raw_results = searcher.pro_search(query, k=k)
    return [
        {
            "category": category,
            "row_idx": row_idx,
            "score": 1,  # Dummy score for pro_search
            "doc": doc
        }
        for doc, category, row_idx in raw_results
    ]


def merge_results(results: List[dict], pro_results: List[dict], limit: int = 5) -> List[dict]:
    """Merge two result lists, keeping the best-scoring entry per row_idx."""
    unique_results = {}
    for res in results + pro_results:
        row_idx = res['row_idx']
        if row_idx not in unique_results or unique_results[row_idx]['score'] < res['score']:
            unique_results[row_idx] = res
    sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
    return sorted_results[:limit]


def parse_search_message(message: str):
    """Extract ``(query, results_json)`` from a search_agent message.

    Accepts the ``For query "...", raw results: ...`` format produced by
    `perform_search`, and falls back to a JSON payload with `query` and
    `results` keys. Returns ``None`` when neither form matches.
    """
    match = re.search(r'For query "(.*?)", raw results: (.*)', message)
    if match:
        return match.group(1), match.group(2)

    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and 'query' in data and 'results' in data:
        return data['query'], json.dumps(data['results'], ensure_ascii=False)
    return None


def build_agent_tools(
    searcher: KeywordSearcher,
    context_vars: ContextVariables,
    score_agent: ConversableAgent,
    custom_search_agent: ConversableAgent,
) -> Tuple[Callable, Callable]:
    """Create the `perform_search` / `handle_results` tools for a given team."""

    def perform_search(query: str, context: ContextVariables):
        """
        Function for search_agent: calls search_products and constructs a message
        with the query and raw results as a JSON string. Updates context_vars
        with the number of results found.
        """
        results = search_products(searcher, query)
        context_vars.data['num_results'] = len(results)
        results_str = json.dumps(results, ensure_ascii=False)
        message = f'For query "{query}", raw results: {results_str}'
        return message

    def handle_results(message: str, context: ContextVariables) -> ReplyResult:
        """
        Tool for check_agent: decides the next step based on the number of results
        stored in context_vars.data['num_results'] and returns a ReplyResult with the
        appropriate message and target agent.
        """
        num_results = context_vars.data['num_results']

        parsed = parse_search_message(message)
        if parsed is None:
            return ReplyResult(
                message="Error: Could not parse the message.",
                target=None,
                context_variables=context_vars
            )
        query, results_str = parsed

        if num_results == 5:
            return ReplyResult(
                message=f"To score_agent: Here are the results: {results_str}",
                target=AgentTarget(score_agent),
                context_variables=context_vars
            )
        elif 1 <= num_results <= 4:
            pro_results = pro_search_products(searcher, query, k=10)
            top5 = merge_results(json.loads(results_str), pro_results)
            top5_str = json.dumps(top5, ensure_ascii=False)
            return ReplyResult(
                message=f"To score_agent: Here are the combined results: {top5_str}",
                target=AgentTarget(score_agent),
                context_variables=context_vars
            )
        else:  # num_results == 0
            return ReplyResult(
                message=f"To custom_search_agent: Please modify the query: {query}",
                target=AgentTarget(custom_search_agent),
                context_variables=context_vars
            )

    return perform_search, handle_results
