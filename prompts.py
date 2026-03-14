"""Prompt variants for customer support ticket classification."""

CATEGORIES = ["billing", "technical", "account", "feature_request", "general"]
CATEGORIES_STR = ", ".join(CATEGORIES)


def prompt_v1_basic(text: str) -> dict:
    """Simple direct instruction."""
    return {
        "system": None,
        "user": (
            f"Classify the following customer support ticket into exactly one of "
            f"these categories: {CATEGORIES_STR}\n\n"
            f"Ticket: {text}\n\n"
            f"Respond with only the category name."
        ),
    }


def prompt_v2_examples(text: str) -> dict:
    """Few-shot with 2 examples per category."""
    examples = (
        "Examples:\n"
        '- "I was overcharged on my last invoice." → billing\n'
        '- "Can I switch to annual billing?" → billing\n'
        '- "The app crashes when I open settings." → technical\n'
        '- "API response times are very slow today." → technical\n'
        '- "I can\'t reset my password." → account\n'
        '- "Please add a new user to our team." → account\n'
        '- "It would be nice to have a dark mode." → feature_request\n'
        '- "Can you add support for webhooks?" → feature_request\n'
        '- "Thanks for the quick help!" → general\n'
        '- "Where can I find your documentation?" → general\n'
    )
    return {
        "system": None,
        "user": (
            f"Classify the following customer support ticket into exactly one of "
            f"these categories: {CATEGORIES_STR}\n\n"
            f"{examples}\n"
            f"Ticket: {text}\n\n"
            f"Respond with only the category name."
        ),
    }


def prompt_v3_cot(text: str) -> dict:
    """Chain-of-thought reasoning before answering."""
    return {
        "system": None,
        "user": (
            f"Classify the following customer support ticket into exactly one of "
            f"these categories: {CATEGORIES_STR}\n\n"
            f"Ticket: {text}\n\n"
            f"Think step by step:\n"
            f"1. What is the customer's core issue or intent?\n"
            f"2. Which category best matches that intent?\n"
            f"3. State your final answer.\n\n"
            f"After your reasoning, write your final answer on a new line in the "
            f'format: ANSWER: <category>'
        ),
    }


def prompt_v4_structured(text: str) -> dict:
    """Strict JSON output format."""
    return {
        "system": None,
        "user": (
            f"Classify the following customer support ticket into exactly one of "
            f"these categories: {CATEGORIES_STR}\n\n"
            f"Ticket: {text}\n\n"
            f'Respond with ONLY a JSON object in this exact format, nothing else:\n'
            f'{{"category": "<category_name>"}}'
        ),
    }


def prompt_v5_persona(text: str) -> dict:
    """System prompt with expert persona."""
    return {
        "system": (
            "You are an expert customer support ticket routing agent with 10 years "
            "of experience. You classify tickets accurately and consistently. "
            "You always respond with exactly one category label and nothing else. "
            f"Valid categories: {CATEGORIES_STR}. "
            "Rules:\n"
            "- billing: payment, invoices, subscriptions, pricing, refunds, charges\n"
            "- technical: bugs, errors, crashes, performance, broken features\n"
            "- account: login, password, profile, permissions, users, SSO, 2FA\n"
            "- feature_request: suggestions, new features, enhancements, wishlists\n"
            "- general: greetings, thanks, questions about the company, anything else"
        ),
        "user": f"Classify this ticket: {text}",
    }


def prompt_v6_tot(text: str) -> dict:
    """Tree of Thought — explore multiple reasoning branches before deciding."""
    return {
        "system": None,
        "user": (
            f"Classify the following customer support ticket into exactly one of "
            f"these categories: {CATEGORIES_STR}\n\n"
            f"Ticket: {text}\n\n"
            f"Use a Tree of Thought approach:\n\n"
            f"Branch A — Read literally: What does the ticket explicitly ask about?\n"
            f"  → Candidate category: ?\n\n"
            f"Branch B — Read by intent: What underlying problem is the customer trying to solve?\n"
            f"  → Candidate category: ?\n\n"
            f"Branch C — Read by elimination: Which categories clearly do NOT fit, and why?\n"
            f"  → Remaining candidate: ?\n\n"
            f"Evaluate branches: Which branch's reasoning is most reliable for this ticket?\n\n"
            f"After evaluating all branches, write your final answer on a new line in the "
            f"format: ANSWER: <category>"
        ),
    }


# Registry of all prompt variants for easy iteration.
PROMPT_VARIANTS = {
    "prompt_v1_basic": prompt_v1_basic,
    "prompt_v2_examples": prompt_v2_examples,
    "prompt_v3_cot": prompt_v3_cot,
    "prompt_v4_structured": prompt_v4_structured,
    "prompt_v5_persona": prompt_v5_persona,
    "prompt_v6_tot": prompt_v6_tot,
}
