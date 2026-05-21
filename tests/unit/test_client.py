"""
Unit tests for src/llm/client.py
Mocks LLM calls so no API key is needed.
"""
import json
from unittest.mock import patch, MagicMock
from pydantic import BaseModel


# ---- Tests for chat() ----

@patch("src.llm.client.get_llm")
def test_chat_returns_text(mock_get_llm):
    mock_response = MagicMock()
    mock_response.content = "Hello!"
    mock_response.usage_metadata = {}
    mock_get_llm.return_value.invoke.return_value = mock_response

    from src.llm.client import chat
    result = chat(messages=[{"content": "Hi"}])

    assert result["text"] == "Hello!"


@patch("src.llm.client.get_llm")
def test_chat_returns_usage(mock_get_llm):
    mock_response = MagicMock()
    mock_response.content = "Hello!"
    mock_response.usage_metadata = {"input_tokens": 10}
    mock_get_llm.return_value.invoke.return_value = mock_response

    from src.llm.client import chat
    result = chat(messages=[{"content": "Hi"}])

    assert "usage" in result


@patch("src.llm.client.get_llm")
def test_chat_logs_call(mock_get_llm, tmp_path, monkeypatch):
    mock_response = MagicMock()
    mock_response.content = "Hello!"
    mock_response.usage_metadata = {}
    mock_get_llm.return_value.invoke.return_value = mock_response

    monkeypatch.setattr("src.llm.client.LOGS_DIR", str(tmp_path))

    from src.llm.client import chat
    chat(messages=[{"content": "Hi"}])

    log_file = tmp_path / "llm_calls.jsonl"
    assert log_file.exists()

    with open(log_file) as f:
        entry = json.loads(f.readline())
    assert "timestamp" in entry
    assert "latency_s" in entry
    assert "response_preview" in entry


@patch("src.llm.client.get_llm")
def test_chat_with_system_prompt(mock_get_llm):
    mock_response = MagicMock()
    mock_response.content = "Response"
    mock_response.usage_metadata = {}
    mock_get_llm.return_value.invoke.return_value = mock_response

    from src.llm.client import chat
    result = chat(
        messages=[{"content": "Hi"}],
        system="You are a helpful assistant"
    )

    assert "text" in result
    assert result["text"] == "Response"


# ---- Tests for parse() ----

class MockSchema(BaseModel):
    answer: str


@patch("src.llm.client.completion")
def test_parse_returns_pydantic_instance(mock_completion):
    mock_choice = MagicMock()
    mock_choice.message.content = '{"answer": "test"}'
    mock_completion.return_value.choices = [mock_choice]

    from src.llm.client import parse
    result = parse(
        messages=[{"content": "What is 2+2?"}],
        response_format=MockSchema,
    )

    assert isinstance(result, MockSchema)
    assert result.answer == "test"


@patch("src.llm.client.completion")
def test_parse_logs_call(mock_completion, tmp_path, monkeypatch):
    mock_choice = MagicMock()
    mock_choice.message.content = '{"answer": "test"}'
    mock_completion.return_value.choices = [mock_choice]

    monkeypatch.setattr("src.llm.client.LOGS_DIR", str(tmp_path))

    from src.llm.client import parse
    parse(messages=[{"content": "Hi"}], response_format=MockSchema)

    log_file = tmp_path / "llm_calls.jsonl"
    assert log_file.exists()

    with open(log_file) as f:
        entry = json.loads(f.readline())
    assert "response_format" in entry
    assert entry["response_format"] == "MockSchema"


@patch("src.llm.client.completion")
def test_parse_with_system_prompt(mock_completion):
    mock_choice = MagicMock()
    mock_choice.message.content = '{"answer": "42"}'
    mock_completion.return_value.choices = [mock_choice]

    from src.llm.client import parse
    result = parse(
        messages=[{"content": "What is 6x7?"}],
        response_format=MockSchema,
        system="You are a math assistant"
    )

    assert result.answer == "42"