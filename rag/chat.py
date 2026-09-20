"""Operator chat entry point — answers a free-text safety question without needing
an incident. This is what backend/'s /chat endpoint will call into.

    python chat.py
"""
from generate import generate_chat_answer


def main():
    print("Campus Sentinel knowledge assistant. Type a question, or 'quit' to exit.\n")
    while True:
        question = input("> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if not question:
            continue

        result = generate_chat_answer(question)
        print(f"\n{result['answer']}")
        print(f"[source: {result['source_section']}]\n")


if __name__ == "__main__":
    main()
