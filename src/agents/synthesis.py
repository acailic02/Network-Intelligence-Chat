from src.llm.client import chat

SYNTHESIS_SYSTEM_PROMPT = """You are an assistant that helps users search through their team's combined LinkedIn network.

CRITICAL RULES — must be strictly followed:

1. GROUNDING: Every claim in your response must be directly based on the profiles you received.
   Never invent information that is not in the profiles.

2. CITATIONS: Cite each person in this format: Full Name profile: URL (connection: Owner)
   Example: Marko Petrovic profile: https://linkedin.com/in/... (connection: Aleksandar)

3. OWNERS: The connection owner must be stated exactly as given in the metadata — never guess.
   If a person has multiple owners, explicitly highlight it: (connection: Aleksandar and Mihajlo)
   Multiple owners = stronger connection, make sure to point this out to the user.

4. NO HALLUCINATIONS: If something is not in the profiles, do not mention it.
   If unsure, say "based on available data".

5. TRANSPARENCY: At the end of your response, if the system did not find everything the user asked for, explicitly state what is missing.

CRITICAL: The "Connection owners" field in each profile shows EXACTLY who knows that person.
Never state an owner that is not listed in "Connection owners".
Even if the user asked for "Petar's connections", if Petar is not in the owners list, do NOT say he is the owner.
Instead, note that this profile was retrieved but Petar is not listed as an owner.

Response format:
- Start with the number of relevant profiles found
- For each person: name, current position, why they are relevant, connection owner
- At the end: strategic recommendation on who is the strongest connection and why (only if clearly supported by the data)
"""


def format_profiles_for_prompt(profiles: list[dict]) -> str:
    """Formats profiles into text the LLM can read.
    Compatible with both structured_filter and semantic_search output formats.
    """
    if not profiles:
        return "No relevant profiles found."

    lines = []
    for i, p in enumerate(profiles, 1):
        owners = p.get("owners", [])
        owners_str = " and ".join(owners) if owners else "UNKNOWN"

        lines.append(f"--- Profile {i} ---")
        lines.append(f"Name: {p.get('name', 'N/A')}")
        lines.append(f"LinkedIn URL: {p.get('linkedin_url', 'N/A')}")
        lines.append(f"Connection owners (VERIFIED, DO NOT CHANGE): {owners_str}")

        # Compatible with both tool formats
        title = p.get("current_job_title") or p.get("current_title", "N/A")
        company = p.get("current_company", "N/A")
        lines.append(f"Current position: {title} @ {company}")

        location = p.get("location")
        if location:
            lines.append(f"Location: {location}")

        industry = p.get("industry")
        if industry:
            lines.append(f"Industry: {industry}")

        headline = p.get("headline")
        if headline:
            lines.append(f"Headline: {headline}")

        summary = p.get("summary")
        if summary:
            lines.append(f"Summary: {summary}")

        skills = p.get("skills", [])
        if skills:
            lines.append(f"Skills: {', '.join(skills)}")

        # Positions from structured_filter
        positions = p.get("positions", [])
        if positions:
            exp_strs = [
                f"{e.get('title')} @ {e.get('company_name')}"
                for e in positions[:3]
                if e.get("title") or e.get("company_name")
            ]
            if exp_strs:
                lines.append(f"Experience: {' | '.join(exp_strs)}")

        # Education from structured_filter
        education = p.get("education", [])
        if education:
            edu_strs = [
                f"{e.get('degree')} @ {e.get('school_name')}"
                for e in education[:2]
                if e.get("degree") or e.get("school_name")
            ]
            if edu_strs:
                lines.append(f"Education: {' | '.join(edu_strs)}")

        lines.append("")

    return "\n".join(lines)

def synthesize(query: str, profiles: list[dict], conversation_history: list[dict] = None,) -> str:
    if not profiles:
        return "No connections found matching your query. Try different keywords."

    for profile in profiles:
        owners = profile.get("owners", [])
        profile["_owners_verified"] = owners

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