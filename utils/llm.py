import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()


def call_llm(prompt: str,
             system: str = "You are a helpful AI assistant.",
             max_tokens: int = 1024) -> str:
    """
    Central wrapper for all LLM calls in ADA.
    All agents call this function — so switching
    between Claude, GPT, or any other model only
    requires changing this ONE function.

    This is called the 'abstraction layer' pattern —
    your agents don't know or care which LLM is used.
    They just call call_llm() and get a response.
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text
