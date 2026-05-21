"""
Agent 3 – Synthesis
Input: top N profiles (with ownership metadata), original query, conversation history.
Output: natural language response with inline citations and connection owners.
"""

from src.llm.client import chat


SYNTHESIS_SYSTEM_PROMPT = """You are an assistant that helps users search through their team's combined LinkedIn network.

CRITICAL RULES — must be strictly followed:

1. GROUNDING: Every claim in your response must be directly based on the profiles you received.
   Never invent information that is not in the profiles.

2. CITATIONS: When mentioning someone, use this format:
   <a href="FULL_LINKEDIN_URL">Full Name</a> (connection: Owner)
   Example: <a href="https://linkedin.com/in/marko-petrovic">Marko Petrovic</a> (connection: Aleksandar)
   Use the exact LinkedIn URL from the profile data — never shorten or modify it.

3. OWNERS: The connection owner must be stated exactly as given in the metadata — never guess.
   CRITICAL: Never state an owner that is not listed in "Connection owners".
   If a person has multiple owners, explicitly highlight it: (connection: Aleksandar and Mihajlo)
   Multiple owners = stronger connection, make sure to point this out.

4. NO HALLUCINATIONS: If something is not in the profiles, do not mention it.
   If unsure, say "based on available data".

5. TRANSPARENCY: If the system did not find everything the user asked for, explicitly state what is missing.

6. CONVERSATION HISTORY: If conversation history is given, determine if the new user query is a FOLLOW-UP to any old user queries or a NEW INDEPENDENT query.
   - FOLLOW-UP: user refines ("from all of the", "filter out"...) or filters previous results (e.g. "filter out Jelena's", "show me more", "now only from Berlin")
     -> acknowledge the refinement in your response, reference previous context naturally
   - NEW INDEPENDENT query: user starts a completely new search with no reference to history
     -> treat as fresh, do not reference previous results
   Conversation history is sorted from oldest to newes messages (top to bottom).
   When in doubt, treat as a NEW INDEPENDENT query.

RESPONSE FORMAT — follow this exactly:
- Start with one sentence stating the total number of relevant profiles found
- Then write ONLY a strategic recommendation: who are the strongest 2-4 connections and why
- Mention only the most relevant profiles using the HTML link format from rule 2
- Do NOT list all profiles one by one with details — those are shown separately below the response
- Keep the response concise — 3-5 sentences maximum
- End with a "Missing:" section only if something was clearly not found

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
        title = p.get("current_job_title") or p.get("current_title")
        company = p.get("current_company")
        if title or company:
            lines.append(f"Current position: {title or 'unknown'} @ {company or 'unknown'}")

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

        # Relevance evaluation from retrieval agent
        relevance = p.get("relevance_summary")
        if relevance:
            lines.append(f"Why relevant: {relevance}")

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


def synthesize(
    query: str,
    profiles: list[dict],
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Generates a natural language response based on retrieved profiles.

    query: original user query
    profiles: list of profiles with ownership metadata
    conversation_history: previous messages in the conversation (optional)
    """
    if not profiles:
        result = ("No connections found matching your query. Try different keywords.", [])
        return result

    filtered_profiles = [p for p in profiles if p.get("relevance_score", 1.0) >= 0.4]

    if not filtered_profiles:
        return ("No sufficiently relevant connections found matching your query. Try different keywords.", [])

    profiles_text = format_profiles_for_prompt(filtered_profiles[:5])

    user_message = f"""User query: {query}

    TOTAL relevant profiles found: {len(filtered_profiles)} (this is exact number)

    TOP 5 best fitted profiles in the team's combined network:

    {profiles_text}

    Based on these profiles, generate a concise strategic response following the format in your instructions.
    Do NOT list all profiles — only highlight the 2-4 strongest ones with HTML links.
    Strictly stick to the data in the profiles — do not invent anything that is not written above."""

    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": "user", "content": msg["user_msg"]})
            messages.append({"role": "assistant", "content": msg["system_res"]})
    messages.append({"role": "user", "content": user_message})

    response = chat(
        messages=messages,
        system=SYNTHESIS_SYSTEM_PROMPT,
    )

    return response["text"], filtered_profiles