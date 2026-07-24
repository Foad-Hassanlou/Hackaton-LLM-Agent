"""Assembly of the agent group chat."""

from dataclasses import dataclass
from typing import List

from autogen import ConversableAgent
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group import AgentTarget, ContextVariables
from autogen.agentchat.group.patterns import AutoPattern

from hackaton import config
from hackaton.agents import prompts
from hackaton.agents.tools import build_agent_tools
from hackaton.search import KeywordSearcher


@dataclass
class AgentTeam:
    """The configured group chat plus the shared context it runs against."""

    pattern: AutoPattern
    context_variables: ContextVariables

    def run(self, user_message: str, max_rounds: int = config.MAX_ROUNDS) -> List[dict]:
        """Run one conversation and return its chat history."""
        result, final_context, last_agent = initiate_group_chat(
            pattern=self.pattern,
            messages=[f"User: {user_message}"],
            max_rounds=max_rounds
        )
        return result.chat_history


def build_agent_team(searcher: KeywordSearcher) -> AgentTeam:
    """Create the agents, wire their tools and hand-offs, and build the pattern.

    The tool-free agents are created first because `build_agent_tools` needs
    `score_agent` and `custom_search_agent` as hand-off targets.
    """
    # Shared context with initial stage and placeholder for number of results
    context_vars = ContextVariables(data={"stage": "search", "num_results": 0})

    llm_config = config.build_llm_config()
    with llm_config:
        user_agent = ConversableAgent(name="user", human_input_mode="ALWAYS")

        custom_search_agent = ConversableAgent(
            name="custom_search_agent",
            system_message=prompts.CUSTOM_SEARCH_AGENT,
        )

        score_agent = ConversableAgent(
            name="score_agent",
            system_message=prompts.SCORE_AGENT,
        )

        perform_search, handle_results = build_agent_tools(
            searcher=searcher,
            context_vars=context_vars,
            score_agent=score_agent,
            custom_search_agent=custom_search_agent,
        )

        search_agent = ConversableAgent(
            name="search_agent",
            system_message=prompts.SEARCH_AGENT,
            functions=[perform_search]
        )

        check_agent = ConversableAgent(
            name="check_agent",
            system_message=prompts.CHECK_AGENT,
            functions=[handle_results]
        )

    search_agent.handoffs.set_after_work(AgentTarget(check_agent))

    # Build the conversation pattern with all agents
    pattern = AutoPattern(
        initial_agent=search_agent,
        agents=[search_agent, check_agent, custom_search_agent, score_agent],
        user_agent=user_agent,
        context_variables=context_vars,
        group_manager_args={"llm_config": llm_config}
    )

    return AgentTeam(pattern=pattern, context_variables=context_vars)
