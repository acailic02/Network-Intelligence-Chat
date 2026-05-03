from src.llm.client import chat

SYNTHESIS_SYSTEM_PROMPT = """You are an assistant that helps users search through their team's combined LinkedIn network.

CRITICAL RULES — must be strictly followed:

1. GROUNDING: Every claim in your response must be directly based on the profiles you received.
   Never invent information that is not in the profiles.

2. CITATIONS: Cite each person in this format: Full Name [profile: URL] (connection: Owner)
   Example: Marko Petrovic [profile: https://linkedin.com/in/...] (connection: Aleksandar)

3. OWNERS: The connection owner must be stated exactly as given in the metadata — never guess.
   If a person has multiple owners, explicitly highlight it: (connection: Aleksandar and Mihajlo)
   Multiple owners = stronger connection, make sure to point this out to the user.

4. NO HALLUCINATIONS: If something is not in the profiles, do not mention it.
   If unsure, say "based on available data".

5. TRANSPARENCY: At the end of your response, if the system did not find everything the user asked for, explicitly state what is missing.

Response format:
- Start with the number of relevant profiles found
- For each person: name, current position, why they are relevant, connection owner
- At the end: strategic recommendation on who is the strongest connection and why (only if clearly supported by the data)
"""


def format_profiles_for_prompt(profiles: list[dict]) -> str:
    if not profiles:
        return "No relevant profiles found."

    lines = []
    for i, p in enumerate(profiles, 1):
        owners = p.get("owners", [])
        owners_str = " and ".join(owners)

        lines.append(f"--- Profile {i} ---")
        lines.append(f"Name: {p.get('name', 'N/A')}")
        lines.append(f"LinkedIn URL: {p.get('linkedin_url', '')}")
        lines.append(f"Connection owners: {owners_str}")
        lines.append(f"Current position: {p.get('current_title', 'N/A')} @ {p.get('current_company', 'N/A')}")
        lines.append(f"Location: {p.get('location', 'N/A')}")
        lines.append(f"Industry: {p.get('industry', 'N/A')}")
        lines.append(f"Headline: {p.get('headline', 'N/A')}")
        lines.append(f"Summary: {p.get('summary', 'N/A')}")

        skills = p.get("skills", [])
        if skills:
            lines.append(f"Skills: {', '.join(skills)}")

        experience = p.get("experience", [])
        if experience:
            exp_strs = []
            for e in experience[:3]:  # max 3 experiences
                end = e.get("end") or "present"
                exp_strs.append(f"{e.get('title')} @ {e.get('company')} ({e.get('start')}–{end})")
            lines.append(f"Experience: {' | '.join(exp_strs)}")

        lines.append("")

    return "\n".join(lines)


def synthesize(query: str, profiles: list[dict], conversation_history: list[dict] = None,) -> str:
    if not profiles:
        return "No connections found matching your query. Try different keywords."

    profiles_text = format_profiles_for_prompt(profiles)

    user_message = f"""User query: {query}

Profiles found in the team's combined network:

{profiles_text}

Based on these profiles, generate a natural response with inline citations and connection owners.
Strictly stick to the data in the profiles — do not invent anything that is not written above."""

    messages = []
    if conversation_history:
        for msg in conversation_history[-4:]:  # max 4 previous messages
            messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    response = chat(
        messages=messages,
        system=SYNTHESIS_SYSTEM_PROMPT,
    )

    return response["text"]