import logging

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s", force=True
)
logger = logging.getLogger(__name__)


from langgraph.graph import StateGraph, START, END
from langgraph.types import StateSnapshot
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages

from typing import TypedDict, Annotated, Literal
import json
import operator
import os
import uuid
from langchain.messages import AIMessage, AnyMessage, HumanMessage
import numpy as np
from agent_registry import AGENT_REGISTRY, DEFAULT_AGENT
from utils import embed, l2_normalize

# Minimum cosine similarity to select a specialist. Override with
# ROUTER_SCORE_THRESHOLD. Short queries often score 0.2–0.4 against the
# canned examples; paraphrases of those examples score much higher.
SCORE_THRESHOLD = float(os.getenv("ROUTER_SCORE_THRESHOLD", "0.3"))
TOP_K = 3

POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://langgraph:langgraph@localhost:5432/langgraph?sslmode=disable",
)

_mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if _mlflow_uri:
    import mlflow

    mlflow.set_tracking_uri(_mlflow_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "langgraph-router"))
    mlflow.langchain.autolog()


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_roles: list[str]
    authorized_agents: list[str]
    selected_agents: list[str]
    agent_scores: dict[str, float]


def authorize_agents(user_roles: list[str]) -> list[str]:
    # Role pre-filter. In the CLI this is a no-op because USER_ROLES lists
    # every role; pass a subset to see agents dropped.
    authorized_agents = [
        agent.agent_id
        for agent in AGENT_REGISTRY.values()
        if agent.required_role == "all" or agent.required_role in user_roles
    ]
    logger.info("Found %d authorized agents", len(authorized_agents))
    return authorized_agents


def score_agents(
    query: str,
    authorized_agents: list[str],
    threshold: float = SCORE_THRESHOLD,
    top_k: int = TOP_K,
) -> tuple[list[str], dict[str, float]]:
    # L2-normalize so cosine similarity (dot product) is in [0, 1]
    query_vector = l2_normalize(embed(query))

    # Score authorized agents (taking max over example questions)
    scored_agents = [
        (agent_id, AGENT_REGISTRY[agent_id].max_similarity(query_vector))
        for agent_id in authorized_agents
    ]

    # Sort descending by similarity
    scored_agents.sort(key=lambda x: x[1], reverse=True)

    # Select multiple agents that meet criteria
    selected = [
        agent_id for agent_id, score in scored_agents[:top_k] if score >= threshold
    ]
    scores = {agent_id: score for agent_id, score in scored_agents}
    logger.info("Selected agents: %s", selected)
    return selected, scores


def route(state: MessagesState):
    user_query = state["messages"][-1].content

    selected_agents, agent_scores = score_agents(
        user_query,
        authorized_agents=state["authorized_agents"],
        threshold=SCORE_THRESHOLD,
        top_k=TOP_K,
    )
    logger.info("router")
    return {"selected_agents": selected_agents, "agent_scores": agent_scores}


def call_agents(state: MessagesState):
    logger.info("Calling %d agents", len(state["selected_agents"]))
    messages = [
        AIMessage(
            content=AGENT_REGISTRY[agent_id].call(state["messages"]),
            name=agent_id,
        )
        for agent_id in state["selected_agents"]
    ]
    return {"messages": messages}


def call_default_agent(state: MessagesState):
    logger.info(
        "Routing to default agent (selected_agents=%s)", state["selected_agents"]
    )
    return {
        "messages": [
            AIMessage(
                content=DEFAULT_AGENT.call(state["messages"]),
                name=DEFAULT_AGENT.agent_id,
            )
        ]
    }


def select_agent_node(
    state: MessagesState,
) -> Literal["call_agents", "call_default_agent"]:
    if state["selected_agents"]:
        return "call_agents"
    return "call_default_agent"


def after_specialists(state: MessagesState) -> Literal["call_default_agent", "__end__"]:
    if len(state["selected_agents"]) > 1:
        return "call_default_agent"
    return END


def create_router_graph() -> StateGraph:
    router = StateGraph(MessagesState)

    router.add_node("route", route)
    router.add_node("call_agents", call_agents)
    router.add_node("call_default_agent", call_default_agent)

    router.add_edge(START, "route")
    router.add_conditional_edges("route", select_agent_node)
    router.add_conditional_edges(
        "call_agents",
        after_specialists,
        {
            "call_default_agent": "call_default_agent",
            END: END,
        },
    )
    router.add_edge("call_default_agent", END)

    return router


def print_recorded_state(snapshot: StateSnapshot, checkpoint_count: int) -> None:
    if not snapshot.values:
        print("\nNo state recorded in Postgres for this thread.")
        return

    rule = "=" * 80
    print(f"\n{rule}")
    print("Recorded state in Postgres")
    print(rule)
    print(f"thread_id:     {snapshot.config['configurable']['thread_id']}")
    print(f"checkpoint_id: {snapshot.config['configurable']['checkpoint_id']}")
    print(f"created_at:    {snapshot.created_at}")
    print(f"step:          {snapshot.metadata.get('step')}")
    print(f"checkpoints:   {checkpoint_count}")

    values = dict(snapshot.values)
    messages = values.pop("messages", [])
    print(f"\n--- state (excluding messages) {'-' * 49}")
    print(json.dumps(values, indent=2, sort_keys=True, default=str))

    print(f"\n--- messages ({len(messages)}) {'-' * 62}")
    for message in messages:
        message.pretty_print()
    print(rule)


# Stand-in: the CLI grants every role so the auth pre-filter never hides an
# agent. Pass a subset here to see the filter work.



if __name__ == "__main__":

    # Mock authorization
    USER_ROLES = [
    "finance",
    "hr",
    "legal",
    "sales",
    "marketing",
    "procurement",
    "it",
    "operations",
]
    
    authorized_agents = authorize_agents(USER_ROLES)

    with PostgresSaver.from_conn_string(POSTGRES_URI) as checkpointer:
        checkpointer.setup()
        graph = create_router_graph().compile(checkpointer=checkpointer)

        # New thread per CLI run; not a real user/session model.
        CONFIG = {"configurable": {"thread_id": str(uuid.uuid4())}}

        print("Chat started. Type 'quit' or 'exit' to stop.")
        message_count = 0
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_input.lower() in {"quit", "exit"}:
                break
            if not user_input:
                continue

            final_state = graph.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "user_roles": USER_ROLES,
                    "authorized_agents": authorized_agents,
                },
                CONFIG,
            )

            for message in final_state["messages"][message_count:]:
                if isinstance(message, HumanMessage):
                    continue
                print(f"{message.name}: {message.content}")
            message_count = len(final_state["messages"])

        snapshot = graph.get_state(CONFIG)
        checkpoint_count = len(list(graph.get_state_history(CONFIG)))
        print_recorded_state(snapshot, checkpoint_count)
