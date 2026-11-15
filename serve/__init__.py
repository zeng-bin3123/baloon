"""Serving layer: request normalisation and HTTP frontend for the baloon inference engine.

Re-exports every public name from serve.frontend and serve.server so that callers can do
``from serve import ChatTemplate, Server, ...`` directly.
"""
from serve.frontend import (
    ChatTemplate,
    Event,
    OutputParser,
    ToolCall,
    _has_image,
    _image_source,
    _parts_of,
    _text_of,
    anthropic_to_messages,
    budget_effort,
    effort_kwargs,
    images_of,
    openai_to_messages,
    parse_tool_call,
)
from serve.server import (
    ByteTokenizer,
    Detokenizer,
    Engine,
    MockEngine,
    RefEngine,
    Server,
    Service,
    Vision,
    anthropic_collect,
    anthropic_events,
    child_env,
    main,
    make_handler,
    openai_chunks,
    openai_collect,
    serve,
)

__all__ = [
    "ChatTemplate", "Event", "OutputParser", "ToolCall",
    "_has_image", "_image_source", "_parts_of", "_text_of",
    "anthropic_to_messages", "budget_effort", "effort_kwargs",
    "images_of", "openai_to_messages", "parse_tool_call",
    "ByteTokenizer", "Detokenizer", "Engine", "MockEngine",
    "RefEngine", "Server", "Service", "Vision",
    "anthropic_collect", "anthropic_events", "child_env", "main",
    "make_handler", "openai_chunks", "openai_collect", "serve",
]
