import json
import time
import os
from datetime import datetime
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import LLM_MODEL, LOGS_DIR

from litellm import completion

def _log(entry: dict):
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "llm_calls.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_llm(model: str = LLM_MODEL) -> ChatLiteLLM:
    return ChatLiteLLM(model=model)


def chat(
    messages: list[dict],
    system: str = "",
    model: str = LLM_MODEL,
) -> dict:
    """
    Wrapper oko LiteLLM-a kompatibilan sa LangChain ekosistemom.
    Loguje svaki poziv u logs/llm_calls.jsonl.
    """
    llm = get_llm(model)

    lc_messages = []
    if system:
        lc_messages.append(SystemMessage(content=system))
    for msg in messages:
        lc_messages.append(HumanMessage(content=msg["content"]))

    start = time.time()
    response = llm.invoke(lc_messages)
    latency = round(time.time() - start, 3)

    _log({
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "latency_s": latency,
        "system_preview": system[:200] if system else "",
        "last_user_message": messages[-1]["content"][:300] if messages else "",
        "response_preview": response.content[:300],
    })

    return {
        "text": response.content,
        "usage": getattr(response, "usage_metadata", {}),
    }

def parse(messages: list[dict], response_format, system: str = "", model: str = LLM_MODEL):
    
    lc_messages = []
    if system:
        lc_messages.append({"role": "system", "content": system})       #SYSTEM PROMT
    for msg in messages:
        lc_messages.append({"role": "user", "content": msg["content"]}) #USER MESSAGE

    start = time.time()
    response = completion(model=model, messages=lc_messages, response_format=response_format) #CALLING LLM WITH PYDANTC CLASS FORMAT 
    latency = round(time.time() - start, 3)

    parsed = response.choices[0].message.content            #JSON FILE GIVEN BY LLM
    result = response_format.model_validate_json(parsed)    #PARSED INTO PYDANTIC CLASS

    _log({
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "latency_s": latency,
        "system_preview": system[:200] if system else "",
        "last_user_message": lc_messages[-1]["content"][:300],
        "response_format": response_format.__name__,
    })

    return result