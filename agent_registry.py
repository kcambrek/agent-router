"""Stand-in specialist catalog for the router spike.

Real department agents are not wired up. Each entry is a generic LLM call plus
a few example questions used only for embedding-based routing.
"""

import numpy as np
from langchain.messages import AnyMessage, SystemMessage

from models import llm
from utils import embed_documents, l2_normalize
import logging

logger = logging.getLogger(__name__)


class AgentSpec:
    def __init__(
        self,
        agent_id: str,
        required_role: str,
        examples: list[str],
        embeddings: np.ndarray | None = None,
        system_prompt: str | None = None,
    ):
        self.agent_id = agent_id
        self.required_role = required_role
        self.examples = examples
        self.system_prompt = system_prompt or f"You are the {agent_id} agent."
        # Pre-normalized matrix of shape (num_examples, embedding_dim).
        # Importing this module calls the embedding API for every non-empty
        # example list — expected for the spike, not a local-only load.
        if embeddings is None:
            if examples:
                logger.info("Generating embeddings for agent %s", agent_id)
                embeddings = embed_documents(examples)
            else:
                # Default/fallback agent is never scored.
                embeddings = np.zeros((0, 1), dtype=np.float64)
        self.embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float64))

    def max_similarity(self, query_vector: np.ndarray) -> float:
        """Cosine similarity against all examples; returns a score in [0, 1]."""
        scores = self.embeddings @ query_vector
        # Plain float, not np.float64: scores land in graph state and the
        # checkpointer cannot serialize numpy scalars.
        max_score = float(np.max(scores))
        logger.info("Score for %s: %.4f", self.agent_id, max_score)
        return max_score

    def call(self, messages: list[AnyMessage]) -> str:
        # Stand-in: one chat completion with a canned system prompt, not a
        # specialist tool graph or department agent.
        response = llm.invoke(
            [
                SystemMessage(content=self.system_prompt),
                *messages,
            ]
        )
        return str(response.content)


# Stand-in catalog. `examples` are the entire routing signal, not eval fixtures.
agent_metadata = {
    "finance": {
        "required_role": "finance",
        "examples": [
            "Prepare the monthly budget variance report.",
            "What invoices are overdue for payment?",
            "Forecast cash flow for the next quarter.",
            "Explain the travel expense reimbursement policy.",
        ],
    },
    "hr": {
        "required_role": "hr",
        "examples": [
            "How many vacation days do I have left?",
            "Help me write a job description for a product manager.",
            "What is the process for onboarding a new employee?",
            "Summarize the results of the employee engagement survey.",
        ],
    },
    "communication": {
        "required_role": "all",
        "examples": [
            "Draft an internal announcement about the office move.",
            "Write a press release for our new product launch.",
            "Create talking points for the quarterly town hall.",
            "Review this customer email for tone and clarity.",
        ],
    },
    "legal": {
        "required_role": "legal",
        "examples": [
            "Review this vendor contract for unusual terms.",
            "What are our obligations under this NDA?",
            "Summarize the data retention requirements.",
            "Draft standard terms for a consulting agreement.",
        ],
    },
    "sales": {
        "required_role": "sales",
        "examples": [
            "Prepare a proposal for a prospective customer.",
            "Summarize this account's recent sales activity.",
            "Draft a follow-up email after a product demo.",
            "Which opportunities are likely to close this quarter?",
        ],
    },
    "marketing": {
        "required_role": "marketing",
        "examples": [
            "Create a campaign brief for the product launch.",
            "Suggest topics for next month's content calendar.",
            "Analyze the performance of our latest email campaign.",
            "Write social media copy for an upcoming webinar.",
        ],
    },
    "procurement": {
        "required_role": "procurement",
        "examples": [
            "Compare these supplier quotes.",
            "Start a purchase request for new laptops.",
            "When does our software vendor contract renew?",
            "Evaluate this supplier against our selection criteria.",
        ],
    },
    "it_support": {
        "required_role": "it",
        "examples": [
            "I cannot connect to the company VPN.",
            "Request access to the analytics dashboard.",
            "My laptop keeps restarting unexpectedly.",
            "How do I reset my company account password?",
        ],
    },
    "operations": {
        "required_role": "operations",
        "examples": [
            "Create a weekly inventory status report.",
            "Investigate the delay in order fulfillment.",
            "Document the process for handling customer returns.",
            "Find ways to reduce warehouse processing time.",
        ],
    },
}

AGENT_REGISTRY = {
    agent_id: AgentSpec(agent_id, **metadata)
    for agent_id, metadata in agent_metadata.items()
}

# Stand-in fallback / synthesizer. Never scored (no examples). Used when
# nothing clears the threshold, or to merge several specialist replies.
DEFAULT_AGENT = AgentSpec(
    "default",
    "all",
    examples=[],
    system_prompt=(
        "You are a helpful assistant. Specialist agents may have already "
        "answered the user's question earlier in the conversation; use their "
        "answers to give one clear reply."
    ),
)
