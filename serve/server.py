"""HTTP serving layer: byte tokeniser, streaming detokeniser, engine protocol, service and server.

One resident engine behind OpenAI / Anthropic style endpoints. ChatService serialises requests
FIFO and rejects (never truncates) a request whose prompt plus max tokens exceed the context.
RefEngine speaks the GEN / T / DONE / ERR protocol. The image-embedding cache holds at most 64
entries (oldest evicted first). With no API key every request is authorised. Ported from
include/baloon/serve_http.hpp.
"""
from __future__ import annotations

import abc
import json
import os
import sys
import threading
from collections import deque
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from serve.frontend import (
    ChatTemplate, Event, OutputParser, ToolCall,
    _image_source, _parts_of, TemplateKwargs,
    anthropic_to_messages, effort_kwargs, images_of, openai_to_messages,
    parse_tool_call,
)

# Token sentinels (mirror of kEos / kImagePad in serve_http.hpp)
_EOS = 256       # end-of-turn token id
_IMAGE_PAD = 257  # <|image_pad|> token id


class ByteTokenizer:
    """Byte-level tokeniser and detokeniser (ByteCodec in serve_http.hpp).

    Every character of a string maps to its byte value (0-255); the special ids _EOS (256) and
    _IMAGE_PAD (257) are written as a unit. encode_with_pad maps every literal occurrence of the
    pad string to _IMAGE_PAD so that vision tokens can be expanded later.
    """

    EOS = _EOS
    IMAGE_PAD = _IMAGE_PAD

    @staticmethod
    def encode(s: str) -> List[int]:
        """Encode a string to a list of byte ids (0-255)."""
        return list(s.encode("utf-8"))

    @staticmethod
    def decode(ids: Iterable[int]) -> str:
        """Decode byte ids to a string, skipping ids outside 0-255."""
        return bytes(t for t in ids if 0 <= t < 256).decode("utf-8", errors="replace")

    @staticmethod
    def encode_with_pad(s: str, pad: str) -> List[int]:
        """Encode s, mapping every literal pad substring to _IMAGE_PAD."""
        if not pad:
            return ByteTokenizer.encode(s)
        out: List[int] = []
        pos = 0
        while pos <= len(s):
            p = s.find(pad, pos)
            if p == -1:
                break
            out.extend(s[pos:p].encode("utf-8"))
            out.append(_IMAGE_PAD)
            pos = p + len(pad)
        out.extend(s[pos:].encode("utf-8"))
        return out


def _utf8_complete_prefix(b: bytes) -> int:
    """Return the length of the longest prefix of b that does not end inside a UTF-8 sequence."""
    n = len(b)
    for back in range(1, min(4, n) + 1):
        c = b[n - back]
        if (c & 0xC0) == 0x80:
            continue
        need = 4 if c >= 0xF0 else (3 if c >= 0xE0 else (2 if c >= 0xC0 else 1))
        return n - back if need > back else n
    return n


class Detokenizer:
    """Streaming UTF-8-safe detokeniser (StreamDecoder in serve_http.hpp).

    push() accumulates byte tokens and returns only the text that became complete (no partial
    multi-byte sequences). The deltas concatenate to the full decoded text. flush() releases any
    remaining bytes, including an incomplete sequence at the end.
    """

    def __init__(self) -> None:
        self._bytes = bytearray()
        self._sent = 0

    @property
    def sent(self) -> int:
        return self._sent

    @property
    def decoded(self) -> int:
        return len(self._bytes)

    def push(self, tok: int) -> str:
        """Accept one token byte and return the newly complete UTF-8 text (may be empty)."""
        if tok < 256:
            self._bytes.append(tok)
        return self._take(_utf8_complete_prefix(self._bytes))

    def flush(self) -> str:
        """Release all remaining accumulated bytes, including any incomplete sequence."""
        return self._take(len(self._bytes))

    def _take(self, cut: int) -> str:
        if cut <= self._sent:
            return ""
        delta = self._bytes[self._sent:cut].decode("utf-8", errors="replace")
        self._sent = cut
        return delta


class Engine(abc.ABC):
    """Abstract generation engine (GenEngine in serve_http.hpp).

    Implementors accept a token-id prompt and a max_new limit, deliver tokens via a sink callable,
    and return a result dict with keys: ok (bool), error (str), n (int, tokens delivered including
    the stop token), stopped (bool, natural stop vs limit hit).
    """

    @abc.abstractmethod
    def generate(
        self,
        prompt: List[int],
        max_new: int,
        sink: Callable[[int], None],
    ) -> dict:
        """Generate up to max_new tokens; call sink for each; return a result dict."""


class MockEngine(Engine):
    """Deterministic engine that replays a fixed token script then emits EOS (ScriptedEngine).

    Useful for unit tests and stdio serving with a known reply. Records every generate() call
    and the prompt that was passed so tests can inspect them.
    """

    def __init__(self, script: List[int]) -> None:
        self._script = list(script)
        self.calls: int = 0
        self.last_prompt: List[int] = []

    def generate(
        self,
        prompt: List[int],
        max_new: int,
        sink: Callable[[int], None],
    ) -> dict:
        self.last_prompt = list(prompt)
        self.calls += 1
        if not prompt or max_new == 0:
            return {"ok": False, "error": "empty prompt or max_new == 0", "n": 0, "stopped": False}
        n = 0
        stopped = False
        for i in range(max_new):
            tok = self._script[i] if i < len(self._script) else _EOS
            sink(tok)
            n += 1
            if tok == _EOS:
                stopped = True
                break
        return {"ok": True, "error": "", "n": n, "stopped": stopped}


class RefEngine(Engine):
    """Engine that talks to an external process via the GEN / T / DONE / ERR line protocol.

    Client side of the protocol used by the engine worker process (LineEngine in serve_http.hpp).
    write_line and read_line are callables supplied by the caller (pipe, socket, etc.).
    """

    def __init__(
        self,
        write_line: Callable[[str], bool],
        read_line: Callable[[], Optional[str]],
    ) -> None:
        self._write = write_line
        self._read = read_line

    def generate(
        self,
        prompt: List[int],
        max_new: int,
        sink: Callable[[int], None],
    ) -> dict:
        if not prompt or max_new == 0:
            return {"ok": False, "error": "empty prompt or max_new == 0", "n": 0, "stopped": False}
        cmd = "GEN " + str(max_new) + "".join(" " + str(t) for t in prompt)
        if not self._write(cmd):
            return {"ok": False, "error": "engine write failed", "n": 0, "stopped": False}
        n = 0
        last = 0
        while True:
            line = self._read()
            if line is None:
                return {"ok": False, "error": "engine closed before DONE", "n": n, "stopped": False}
            if line == "DONE":
                break
            if line.startswith("ERR "):
                return {"ok": False, "error": line[4:], "n": n, "stopped": False}
            if not line.startswith("T "):
                return {"ok": False, "error": "bad engine line", "n": n, "stopped": False}
            try:
                tok = int(line[2:])
            except ValueError:
                return {"ok": False, "error": "bad token line", "n": n, "stopped": False}
            if n >= max_new:
                return {"ok": False, "error": "engine exceeded max_new", "n": n, "stopped": False}
            last = tok
            n += 1
            sink(tok)
        stopped = n < max_new or (n > 0 and last == _EOS)
        return {"ok": True, "error": "", "n": n, "stopped": stopped}


_VISION_CACHE_MAX = 64


class Vision:
    """Image-embedding cache: source URL → token count (ImageEmbedder in serve_http.hpp).

    Wraps a user-supplied compute function and caches results in a FIFO bounded at 64 entries
    (oldest evicted first). The compute function is called at most once per unique source.
    """

    def __init__(self, compute: Callable[[str], int]) -> None:
        self._compute = compute
        self._cache: deque = deque()   # deque of (key, value) pairs
        self._hits = 0
        self._misses = 0

    def tokens_for(self, source: str) -> int:
        """Return the token count for an image source, computing and caching on first access."""
        for k, v in self._cache:
            if k == source:
                self._hits += 1
                return v
        self._misses += 1
        n = self._compute(source)
        # Evict oldest if at capacity
        self._cache.append((source, n))
        while len(self._cache) > _VISION_CACHE_MAX:
            self._cache.popleft()
        return n

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses


def _expand_pads(ids: List[int], pad: int, counts: List[int]) -> List[int]:
    """Expand each occurrence of pad to counts[i] pads in sequence; extra occurrences stay as-is."""
    out: List[int] = []
    k = 0
    for t in ids:
        if t == pad and k < len(counts):
            out.extend([pad] * counts[k])
            k += 1
        else:
            out.append(t)
    return out


class Service:
    """FIFO request queue wrapping an Engine (ChatService in serve_http.hpp).

    Serialises requests strictly in arrival order and rejects (never truncates) a request whose
    prompt plus max_new exceed the context. Vision sources are embedded via the Vision cache.
    Returns completion dicts compatible with Server.handle().
    """

    def __init__(self, engine: Engine, vision: Vision, max_ctx: int) -> None:
        self._engine = engine
        self._vision = vision
        self._max_ctx = max_ctx
        self._tpl = ChatTemplate()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._next_ticket = 0
        self._now_serving = 0
        self._served = 0

    @property
    def context(self) -> int:
        return self._max_ctx

    @property
    def served(self) -> int:
        return self._served

    def prepare(self, messages: list, kw: TemplateKwargs, max_new: int) -> dict:
        """Encode messages to token ids and validate context bounds. Returns a prepared dict."""
        if max_new == 0:
            return {"ok": False, "status": 400, "error": "max tokens must be positive",
                    "ids": [], "max_new": 0}
        counts = []
        for src in images_of(messages):
            if not _image_source(src)["ok"]:
                return {"ok": False, "status": 400, "error": "unsupported image source",
                        "ids": [], "max_new": max_new}
            counts.append(self._vision.tokens_for(src))
        ids = ByteTokenizer.encode_with_pad(self._tpl.render(messages, kw), self._tpl.image_pad)
        ids = _expand_pads(ids, _IMAGE_PAD, counts)
        if len(ids) + max_new > self._max_ctx:
            return {
                "ok": False, "status": 400,
                "error": "prompt ({p}) + max tokens ({m}) exceed the context ({c})".format(
                    p=len(ids), m=max_new, c=self._max_ctx),
                "ids": [], "max_new": max_new,
            }
        return {"ok": True, "status": 200, "error": "", "ids": ids, "max_new": max_new}

    def run(self, prepared: dict, start_in_reasoning: bool = True) -> dict:
        """Run a prepared request through the engine; waits for FIFO turn. Returns completion dict."""
        if not prepared.get("ok"):
            return {"ok": False, "status": prepared.get("status", 500),
                    "error": prepared.get("error", ""), "prompt_tokens": 0,
                    "completion_tokens": 0, "stopped": False, "calls": 0, "events": []}
        # FIFO ordering via ticket
        with self._cv:
            my_ticket = self._next_ticket
            self._next_ticket += 1
            self._cv.wait_for(lambda: self._now_serving == my_ticket)

        try:
            dec = Detokenizer()
            parser = OutputParser(start_in_reasoning)
            events: List[Event] = []

            def _merge(new_events):
                for e in new_events:
                    if events and e.kind != Event.Call and events[-1].kind == e.kind:
                        events[-1].text += e.text
                    else:
                        events.append(e)

            def _sink(tok: int) -> None:
                if tok == _EOS:
                    return
                _merge(parser.feed(dec.push(tok)))

            result = self._engine.generate(prepared["ids"], prepared["max_new"], _sink)
            if not result.get("ok"):
                return {"ok": False, "status": 500, "error": result.get("error", ""),
                        "prompt_tokens": 0, "completion_tokens": 0,
                        "stopped": False, "calls": 0, "events": []}
            _merge(parser.feed(dec.flush()))
            _merge(parser.finish())
            return {
                "ok": True, "status": 200, "error": "",
                "prompt_tokens": len(prepared["ids"]),
                "completion_tokens": result["n"],
                "stopped": result.get("stopped", False),
                "calls": parser.calls,
                "events": events,
            }
        finally:
            with self._cv:
                self._now_serving += 1
                self._served += 1
                self._cv.notify_all()


class Server:
    """HTTP-level request handler (HttpServer in serve_http.hpp).

    Routes to OpenAI (/v1/chat/completions) or Anthropic (/v1/messages) endpoints, checks the
    API key when one is configured (empty key → allow all), and returns a response dict with
    keys: status (int), content_type (str), body (str), frames (list[str] for streaming).
    """

    def __init__(self, service: Service, api_key: str = "") -> None:
        self._svc = service
        self._key = api_key

    def handle(self, request: dict) -> dict:
        """Dispatch one ApiRequest dict and return an ApiResponse dict."""
        if self._key and request.get("key", "") != self._key:
            return _api_error(401, "invalid API key")
        route = request.get("route", "")
        is_openai = route == "/v1/chat/completions"
        is_anthropic = route == "/v1/messages"
        if not is_openai and not is_anthropic:
            return _api_error(404, "unknown route")
        if is_openai:
            msgs = openai_to_messages(request.get("messages", []))
        else:
            msgs = anthropic_to_messages(request.get("system", ""), request.get("messages", []))
        if msgs is None:
            return _api_error(400, "unsupported role")
        kw = effort_kwargs(request.get("effort", "medium"))
        if not kw.ok:
            return _api_error(400, "unknown reasoning effort")
        max_new = _pick_max(
            request.get("max_completion_tokens", 0),
            request.get("max_tokens", 0),
        )
        prepared = self._svc.prepare(msgs, kw, max_new)
        if not prepared["ok"]:
            return _api_error(prepared["status"], prepared["error"])
        completion = self._svc.run(prepared, kw.enable_thinking)
        if not completion["ok"]:
            return _api_error(completion["status"], completion["error"])
        req_id = ("chatcmpl-" if is_openai else "msg_") + str(self._svc.served)
        model = request.get("model", "baloon")
        if request.get("stream"):
            frames = openai_chunks(req_id, model, completion) if is_openai else \
                     anthropic_events(req_id, model, completion)
            return {"status": 200, "content_type": "text/event-stream",
                    "body": "", "frames": frames}
        body = openai_collect(req_id, model, completion) if is_openai else \
               anthropic_collect(req_id, model, completion)
        return {"status": 200, "content_type": "application/json", "body": body, "frames": []}


def _api_error(status: int, msg: str) -> dict:
    return {"status": status, "content_type": "application/json",
            "body": json.dumps({"error": {"message": msg}}), "frames": []}


def _pick_max(a: int, b: int) -> int:
    """max_completion_tokens wins over max_tokens; default to 1024 when both are zero."""
    return a if a else (b if b else 1024)


def _json_str(s: str) -> str:
    """JSON-encode a string with the same escaping as the C++ json_str."""
    return json.dumps(s, ensure_ascii=False)


def _args_json(c: ToolCall) -> str:
    """Serialize tool call arguments as a JSON object."""
    return json.dumps(dict(c.args), ensure_ascii=False)


def _openai_finish_name(completion: dict) -> str:
    calls = completion.get("calls", 0)
    stopped = completion.get("stopped", False)
    if stopped and calls > 0:
        return "tool_calls"
    return "stop" if stopped else "length"


def _anthropic_stop_name(completion: dict) -> str:
    calls = completion.get("calls", 0)
    stopped = completion.get("stopped", False)
    if calls > 0 and stopped:
        return "tool_use"
    return "end_turn" if stopped else "max_tokens"


def _usage_json(completion: dict) -> str:
    p = completion.get("prompt_tokens", 0)
    c = completion.get("completion_tokens", 0)
    return '{{"prompt_tokens":{p},"completion_tokens":{c},"total_tokens":{t}}}'.format(
        p=p, c=c, t=p + c)


def anthropic_collect(req_id: str, model: str, completion: dict) -> str:
    """Build the Anthropic non-streaming response body (anthropic_body in serve_http.hpp)."""
    blocks_parts = []
    k = 0
    for e in completion.get("events", []):
        if e.kind == Event.Reasoning:
            blocks_parts.append('{{"type":"thinking","thinking":{t}}}'.format(t=_json_str(e.text)))
        elif e.kind == Event.Content:
            blocks_parts.append('{{"type":"text","text":{t}}}'.format(t=_json_str(e.text)))
        else:
            blocks_parts.append(
                '{{"type":"tool_use","id":"toolu_{k}","name":{n},"input":{a}}}'.format(
                    k=k, n=_json_str(e.call.name), a=_args_json(e.call)))
            k += 1
    p = completion.get("prompt_tokens", 0)
    c = completion.get("completion_tokens", 0)
    return (
        '{{"id":{id},"type":"message","role":"assistant","model":{m},"content":[{b}],'
        '"stop_reason":{sr},"usage":{{"input_tokens":{p},"output_tokens":{c}}}}}'
    ).format(
        id=_json_str(req_id), m=_json_str(model),
        b=",".join(blocks_parts),
        sr=_json_str(_anthropic_stop_name(completion)),
        p=p, c=c,
    )


def anthropic_events(req_id: str, model: str, completion: dict) -> List[str]:
    """Build the Anthropic streaming event list (anthropic_stream in serve_http.hpp).

    Emits message_start, one content_block_start + content_block_delta + content_block_stop per
    event, then message_delta and message_stop.
    """
    out: List[str] = []
    p = completion.get("prompt_tokens", 0)
    c = completion.get("completion_tokens", 0)

    def ev(name: str, data: str) -> None:
        out.append("event: {n}\ndata: {d}\n\n".format(n=name, d=data))

    ev("message_start",
       '{{"type":"message_start","message":{{"id":{id},"type":"message","role":"assistant",'
       '"model":{m},"content":[],"usage":{{"input_tokens":{p},"output_tokens":0}}}}}}'.format(
           id=_json_str(req_id), m=_json_str(model), p=p))

    idx = 0
    k = 0
    for e in completion.get("events", []):
        i = str(idx)
        if e.kind == Event.Call:
            ev("content_block_start",
               '{{"type":"content_block_start","index":{i},"content_block":{{"type":"tool_use",'
               '"id":"toolu_{k}","name":{n},"input":{{}}}}}}'.format(
                   i=i, k=k, n=_json_str(e.call.name)))
            ev("content_block_delta",
               '{{"type":"content_block_delta","index":{i},"delta":{{"type":"input_json_delta",'
               '"partial_json":{a}}}}}'.format(i=i, a=_json_str(_args_json(e.call))))
            k += 1
        else:
            is_think = e.kind == Event.Reasoning
            block_type = "thinking" if is_think else "text"
            block_field = "thinking" if is_think else "text"
            ev("content_block_start",
               '{{"type":"content_block_start","index":{i},"content_block":{{"type":"{bt}","{bf}":""}}}}'.format(
                   i=i, bt=block_type, bf=block_field))
            delta_type = "thinking_delta" if is_think else "text_delta"
            delta_field = "thinking" if is_think else "text"
            ev("content_block_delta",
               '{{"type":"content_block_delta","index":{i},"delta":{{"type":"{dt}","{df}":{t}}}}}'.format(
                   i=i, dt=delta_type, df=delta_field, t=_json_str(e.text)))
        ev("content_block_stop",
           '{{"type":"content_block_stop","index":{i}}}'.format(i=i))
        idx += 1

    ev("message_delta",
       '{{"type":"message_delta","delta":{{"stop_reason":{sr}}},'
       '"usage":{{"output_tokens":{c}}}}}'.format(
           sr=_json_str(_anthropic_stop_name(completion)), c=c))
    ev("message_stop", '{"type":"message_stop"}')
    return out


def child_env(base: List[str], gpus: str) -> List[str]:
    """Return the worker environment with CUDA_VISIBLE_DEVICES replaced (worker_env in C++).

    Strips any existing CUDA_VISIBLE_DEVICES from base, then appends a fresh one when gpus is
    non-empty. The order of the remaining variables is preserved.
    """
    out = [e for e in base if not e.startswith("CUDA_VISIBLE_DEVICES=")]
    if gpus:
        out.append("CUDA_VISIBLE_DEVICES=" + gpus)
    return out


def make_handler(server: Server) -> Callable[[dict], dict]:
    """Return a callable that dispatches ApiRequest dicts through server (build_handler in C++)."""
    return lambda request: server.handle(request)


def openai_chunks(req_id: str, model: str, completion: dict) -> List[str]:
    """Build the OpenAI streaming frame list (openai_stream in serve_http.hpp).

    Emits one SSE data frame per event, preceded by the role delta frame and followed by the
    finish-reason frame and the [DONE] sentinel.
    """
    out: List[str] = []

    def frame(delta: str, finish: str) -> None:
        out.append(
            'data: {{"id":{id},"object":"chat.completion.chunk","model":{m},'
            '"choices":[{{"index":0,"delta":{d},"finish_reason":{f}}}]}}\n\n'.format(
                id=_json_str(req_id), m=_json_str(model), d=delta, f=finish))

    frame('{"role":"assistant"}', "null")
    k = 0
    for e in completion.get("events", []):
        if e.kind == Event.Content:
            frame('{{"content":{t}}}'.format(t=_json_str(e.text)), "null")
        elif e.kind == Event.Reasoning:
            frame('{{"reasoning_content":{t}}}'.format(t=_json_str(e.text)), "null")
        else:
            frame(
                '{{"tool_calls":[{{"index":{k},"id":"call_{k}","type":"function",'
                '"function":{{"name":{n},"arguments":{a}}}}}]}}'.format(
                    k=k, n=_json_str(e.call.name), a=_json_str(_args_json(e.call))),
                "null")
            k += 1
    frame("{}", _json_str(_openai_finish_name(completion)))
    out.append("data: [DONE]\n\n")
    return out


def openai_collect(req_id: str, model: str, completion: dict) -> str:
    """Build the OpenAI non-streaming response body (openai_body in serve_http.hpp)."""
    content = ""
    reasoning = ""
    calls_parts: List[str] = []
    k = 0
    for e in completion.get("events", []):
        if e.kind == Event.Content:
            content += e.text
        elif e.kind == Event.Reasoning:
            reasoning += e.text
        else:
            calls_parts.append(
                '{{"id":"call_{k}","type":"function","function":{{"name":{n},"arguments":{a}}}}}'.format(
                    k=k, n=_json_str(e.call.name), a=_json_str(_args_json(e.call))))
            k += 1
    msg = '{{"role":"assistant","content":{c}'.format(c=_json_str(content))
    if reasoning:
        msg += ',"reasoning_content":{r}'.format(r=_json_str(reasoning))
    if calls_parts:
        msg += ',"tool_calls":[{tc}]'.format(tc=",".join(calls_parts))
    msg += "}"
    return (
        '{{"id":{id},"object":"chat.completion","model":{m},'
        '"choices":[{{"index":0,"message":{msg},"finish_reason":{fr}}}],'
        '"usage":{u}}}'
    ).format(
        id=_json_str(req_id), m=_json_str(model), msg=msg,
        fr=_json_str(_openai_finish_name(completion)),
        u=_usage_json(completion),
    )


def serve(handler: Callable[[dict], dict], requests: List[dict]) -> List[dict]:
    """Serve a batch of requests in arrival order (run_requests in serve_http.hpp).

    Each request dict is dispatched through handler in sequence; the results list preserves order.
    This is the socket-free serving loop used in tests and for stdio operation.
    """
    return [handler(r) for r in requests]


def main(argv: Optional[List[str]] = None) -> int:
    """Stdio front-end: --ctx N --key K --gpus G --script TEXT (serve_main in serve_http.hpp).

    Each stdin line becomes a user turn answered with an OpenAI-format body on stdout.
    Returns 0 on success, 2 on argument errors.
    """
    import sys as _sys
    args = argv if argv is not None else _sys.argv[1:]
    ctx = 4096
    key = ""
    gpus = ""
    script_text = "</think>ok"
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--ctx":
            i += 1
            if i >= len(args) or not args[i].isdigit():
                return 2
            ctx = int(args[i])
            if ctx == 0:
                return 2
        elif a == "--key":
            i += 1
            if i >= len(args):
                return 2
            key = args[i]
        elif a == "--gpus":
            i += 1
            if i >= len(args):
                return 2
            gpus = args[i]
        elif a == "--script":
            i += 1
            if i >= len(args):
                return 2
            script_text = args[i]
        else:
            return 2
        i += 1
    eng = MockEngine(ByteTokenizer.encode(script_text))
    vis = Vision(lambda _src: 16)
    svc = Service(eng, vis, ctx)
    srv = Server(svc, key)
    h = make_handler(srv)
    for line in _sys.stdin:
        line = line.rstrip("\n")
        request = {
            "route": "/v1/chat/completions",
            "key": key,
            "max_tokens": 256,
            "messages": [{"role": "user", "parts": _parts_of(line), "calls": []}],
        }
        resp = h(request)
        print("{status} {body}".format(status=resp["status"], body=resp["body"]))
    return 0
