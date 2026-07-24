"""LCEL RAG chain: retrieve -> prompt -> Groq LLM -> string (FR-10..FR-13).

The system prompt forces grounding in retrieved context only and defines the
escalation behaviour when context is insufficient (FR-10, FR-11).
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

import config
from retriever import format_context, retrieve_context, warm_up

SYSTEM_PROMPT = """\
You are MyTelecom's customer-care assistant. You help subscribers resolve \
Tier-1 support questions about connectivity, data, roaming, SIM/eSIM, billing, \
voice calls, devices, and account/app usage.

STRICT RULES:
1. Answer using ONLY the information in the CONTEXT below. Do not use any prior \
or general knowledge, and never invent prices, policies, codes, or steps.
2. If the CONTEXT does not contain enough information to answer confidently, \
say so plainly and tell the user to call 611 or use the MyTelecom app. Do not \
guess.
3. You have no access to live account data (balances, personal usage, specific \
charges). If asked for personalised account details, explain that you can't see \
their account and direct them to the MyTelecom app or 611.
4. Be concise and practical. Use short numbered steps for troubleshooting.
5. Answer in English.

CONTEXT:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def get_llm() -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatGroq(
        model=config.GROQ_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
        # Strip the model's reasoning trace, keep only the final answer.
        reasoning_format="parsed",
    )


def build_generation_chain():
    """Chain that generates from already-retrieved context.

    Input: {"question": str, "context": str} -> str answer. Used by the UI,
    which retrieves separately so it can also display the source documents.
    """
    warm_up()
    return PROMPT | get_llm() | StrOutputParser()


def build_chain():
    """Self-contained chain: {"question": str} -> str answer.

    Retrieves internally. Used by the CLI, where sources aren't displayed.
    """
    warm_up()
    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: retrieve_context(x["question"]))
        )
        | PROMPT
        | get_llm()
        | StrOutputParser()
    )


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "Why is my mobile internet slow?"
    chain = build_chain()
    for token in chain.stream({"question": question}):
        print(token, end="", flush=True)
    print()
