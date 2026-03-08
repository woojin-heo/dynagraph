from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
LLM = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)

ACTION_LLM_OVERRIDES: dict[str, dict] = {
    "SQL_GENERATION": {"model": "gpt-4o", "temperature": 0},
}


def get_llm_for_action(action_type: str) -> ChatOpenAI:
    """Return the LLM for a given action: override if configured, else default."""
    cfg = ACTION_LLM_OVERRIDES.get(action_type)
    if cfg:
        return ChatOpenAI(**cfg)
    return LLM
