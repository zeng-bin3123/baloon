"""Frontend request normalisation: message conversion, effort mapping, streaming reply parser.

OpenAI / Anthropic messages → unified Message list; reasoning effort maps to template levels
(monotone: <2048 low, <8192 medium, else xhigh). The streaming reply parser splits model output
into reasoning, content and tool calls and never emits a partial tag (held suffix is a proper
prefix of the awaited tag). Tool-call parameter values lose at most one leading and one trailing
newline. Ported from include/baloon/serve_chat.hpp and include/baloon/serve_frontend.hpp.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ToolCall:
    """Named function invocation with ordered key/value arguments (ToolInvocation in C++)."""
    name: str = ""
    args: List[Tuple[str, str]] = field(default_factory=list)


class Event:
    """One output event from the reply parser: reasoning text, content text, or a tool call.

    Mirrors ChatEvent in serve_chat.hpp. kind is one of Event.Reasoning, Event.Content,
    Event.Call. For reasoning/content events, text holds the text; for call events, call holds
    the ToolCall.
    """
    Reasoning = "reasoning"
    Content = "content"
    Call = "call"

    def __init__(self, kind: str = "content", text: str = "", call: Optional[ToolCall] = None) -> None:
        self.kind = kind
        self.text = text
        self.call = call if call is not None else ToolCall()


class ChatTemplate:
    """Renders messages into the prompt string expected by the model.

    One <|image_pad|> per image (the service expands each to one per image token).
    With enable_thinking, opens with a Reasoning effort system turn and appends <think>\n;
    without it, appends the collapsed <think>\n\n</think>\n\n suffix. Mirrors ChatTemplate in
    serve_chat.hpp.
    """

    def __init__(self) -> None:
        self.im_start = "<|im_start|>"
        self.im_end = "<|im_end|>"
        self.image_pad = "<|image_pad|>"
        self.vision_start = "<|vision_start|>"
        self.vision_end = "<|vision_end|>"

    @staticmethod
    def render_call(c: ToolCall) -> str:
        """Render one ToolCall to the wire format used in assistant turns."""
        s = "<tool_call>\n<function={name}>\n".format(name=c.name)
        for k, v in c.args:
            s += "<parameter={k}>\n{v}\n</parameter>\n".format(k=k, v=v)
        return s + "</function>\n</tool_call>"

    def render(self, messages: list, kw: "TemplateKwargs") -> str:
        """Render a message list to the full prompt string including the assistant prefix."""
        s = ""
        if kw.enable_thinking and kw.effort:
            s += self.im_start + "system\nReasoning effort: " + kw.effort + self.im_end + "\n"
        for m in messages:
            s += self.im_start + m["role"] + "\n"
            any_part = False
            for p in m.get("parts", []):
                if p.get("image"):
                    s += self.vision_start + self.image_pad + self.vision_end
                else:
                    s += p.get("text", "")
                any_part = True
            for c in m.get("calls", []):
                if any_part:
                    s += "\n"
                s += self.render_call(c)
                any_part = True
            s += self.im_end + "\n"
        s += self.im_start + "assistant\n"
        s += "<think>\n" if kw.enable_thinking else "<think>\n\n</think>\n\n"
        return s


def _hold(text: str, tag: str) -> int:
    """Return the longest suffix of text that is a proper prefix of tag (never the whole tag)."""
    n = min(len(tag) - 1, len(text))
    for size in range(n, 0, -1):
        if text[-size:] == tag[:size]:
            return size
    return 0


class OutputParser:
    """Incremental streaming reply parser (ReplyParser in serve_chat.hpp).

    Splits model output into reasoning, content and tool call events without ever exposing
    a partial tag. Transitions: Reasoning -> Content (on </think>) -> Call (on <tool_call>)
    -> Content (on </tool_call>); never back to Reasoning.
    """

    _THINK_END = "</think>"
    _CALL_START = "<tool_call>"
    _CALL_END = "</tool_call>"

    def __init__(self, start_in_reasoning: bool = True) -> None:
        self._state = "reasoning" if start_in_reasoning else "content"
        self._buf = ""
        self._call_body = ""
        self._calls = 0
        self._bad_call = False
        self._truncated_call = False

    @property
    def state(self) -> str:
        return self._state

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def bad_call(self) -> bool:
        return self._bad_call

    @property
    def truncated_call(self) -> bool:
        return self._truncated_call

    @property
    def held(self) -> int:
        return len(self._buf)

    def feed(self, chunk: str) -> List[Event]:
        self._buf += chunk
        return self._drain(False)

    def finish(self) -> List[Event]:
        return self._drain(True)

    def _emit(self, out: list, text: str) -> None:
        if not text:
            return
        kind = Event.Reasoning if self._state == "reasoning" else Event.Content
        if out and out[-1].kind == kind and out[-1].kind != Event.Call:
            out[-1].text += text
        else:
            out.append(Event(kind=kind, text=text))

    def _drain(self, fin: bool) -> List[Event]:
        out: List[Event] = []
        while True:
            if self._state == "reasoning":
                tag = self._THINK_END
            elif self._state == "content":
                tag = self._CALL_START
            else:
                tag = self._CALL_END

            pos = self._buf.find(tag)

            if self._state == "call":
                if pos == -1:
                    # Accumulate call body, but hold the tail that might be </tool_call>
                    hold = _hold(self._buf, tag)
                    self._call_body += self._buf[:len(self._buf) - hold]
                    self._buf = self._buf[len(self._buf) - hold:]
                    if fin:
                        self._truncated_call = True
                        self._call_body = ""
                        self._buf = ""
                    return out
                # Complete call body found
                self._call_body += self._buf[:pos]
                self._buf = self._buf[pos + len(tag):]
                c = parse_tool_call(self._call_body)
                if c is not None:
                    ev = Event(kind=Event.Call)
                    ev.call = c
                    out.append(ev)
                    self._calls += 1
                else:
                    self._bad_call = True
                self._call_body = ""
                self._state = "content"
                continue

            if pos != -1:
                self._emit(out, self._buf[:pos])
                self._buf = self._buf[pos + len(tag):]
                if self._state == "reasoning":
                    self._state = "content"
                else:
                    self._state = "call"
                continue

            hold = 0 if fin else _hold(self._buf, tag)
            self._emit(out, self._buf[:len(self._buf) - hold])
            self._buf = self._buf[len(self._buf) - hold:]
            return out


def _has_image(msg: dict) -> bool:
    """Return True if any part of the message carries an image (has_image in serve_chat.hpp)."""
    return any(p.get("image") for p in msg.get("parts", []))


def _image_source(url: str) -> dict:
    """Parse an image URL into {ok, inline_data, payload} (image_source in serve_chat.hpp).

    Accepts data: URLs with ;base64, payloads and http(s) URLs; everything else has ok=False.
    """
    B64_MARK = ";base64,"
    if url.startswith("data:"):
        k = url.find(B64_MARK)
        if k == -1 or k + len(B64_MARK) >= len(url):
            return {"ok": False, "inline_data": False, "payload": ""}
        return {"ok": True, "inline_data": True, "payload": url[k + len(B64_MARK):]}
    if url.startswith("http://") or url.startswith("https://"):
        return {"ok": True, "inline_data": False, "payload": url}
    return {"ok": False, "inline_data": False, "payload": ""}


def _parts_of(content: str) -> list:
    """Convert a plain string to a one-element parts list; empty string yields empty list.

    Mirrors parts_of in serve_chat.hpp.
    """
    return [{"image": False, "text": content}] if content else []


def _text_of(msg: dict) -> str:
    """Concatenate all text parts of a message, skipping image parts (text_of in serve_chat.hpp)."""
    return "".join(p.get("text", "") for p in msg.get("parts", []) if not p.get("image"))


def anthropic_to_messages(system: str, messages: list) -> Optional[list]:
    """Convert Anthropic-dialect messages to the unified list (from_anthropic in serve_chat.hpp).

    The system prompt is a separate field and goes first as a system role message. Only user and
    assistant turns are allowed in messages; any other role returns None (400 to the caller).
    """
    out = []
    if system:
        out.append({"role": "system", "parts": _parts_of(system), "calls": []})
    for m in messages:
        if m.get("role") not in ("user", "assistant"):
            return None
        out.append(m)
    return out


def budget_effort(thinking_budget: int) -> str:
    """Map an Anthropic thinking-budget token count to a level name (budget_effort_name in C++).

    Monotone bucketing: <2048 → "low", <8192 → "medium", else "xhigh".
    """
    if thinking_budget < 2048:
        return "low"
    if thinking_budget < 8192:
        return "medium"
    return "xhigh"


# Internal sentinel to represent a parsed TemplateKwargs.
@dataclass
class TemplateKwargs:
    ok: bool = False
    enable_thinking: bool = True
    effort: str = ""


_EFFORT_DISABLED = frozenset(("none", "off", "minimal", "disabled"))
_EFFORT_LOW = frozenset(("low",))
_EFFORT_MEDIUM = frozenset(("medium",))
_EFFORT_HIGH = frozenset(("high", "xhigh", "max", "maximum"))


def effort_kwargs(effort_spelling: str) -> TemplateKwargs:
    """Parse a reasoning-effort string to TemplateKwargs (template_kwargs in serve_chat.hpp).

    Returns TemplateKwargs with ok=False for unrecognised spellings (the caller returns HTTP 400).
    Disabled spellings set enable_thinking=False and effort="". Known spellings map onto the
    three level names low / medium / xhigh.
    """
    kw = TemplateKwargs()
    if effort_spelling in _EFFORT_DISABLED:
        kw.ok = True
        kw.enable_thinking = False
        kw.effort = ""
        return kw
    if effort_spelling in _EFFORT_LOW:
        kw.ok = True
        kw.effort = "low"
        return kw
    if effort_spelling in _EFFORT_MEDIUM:
        kw.ok = True
        kw.effort = "medium"
        return kw
    if effort_spelling in _EFFORT_HIGH:
        kw.ok = True
        kw.effort = "xhigh"
        return kw
    return kw  # ok stays False


def images_of(messages: list) -> List[str]:
    """Collect image sources from all messages in order of appearance (collect_images in C++)."""
    result = []
    for m in messages:
        for p in m.get("parts", []):
            if p.get("image"):
                result.append(p.get("source", ""))
    return result


def openai_to_messages(messages: list) -> Optional[list]:
    """Convert OpenAI-dialect messages to the unified list (from_openai in serve_chat.hpp).

    developer → system, function → tool; messages with neither parts nor calls are dropped.
    Unknown roles return None (the caller returns HTTP 400).
    """
    allowed = {"system", "user", "assistant", "tool"}
    out = []
    for m in messages:
        m = dict(m)
        role = m.get("role", "")
        if role == "developer":
            m["role"] = "system"
        elif role == "function":
            m["role"] = "tool"
        if m["role"] not in allowed:
            return None
        if not m.get("parts") and not m.get("calls"):
            continue
        out.append(m)
    return out


def _strip_value(v: str) -> str:
    """Remove at most one leading and one trailing newline from a parameter value."""
    if v and v[0] == "\n":
        v = v[1:]
    if v and v[-1] == "\n":
        v = v[:-1]
    return v


def parse_tool_call(body: str) -> Optional[ToolCall]:
    """Parse the body of a <tool_call> block (decode_tool_call in serve_chat.hpp).

    Expects: <function=NAME><parameter=K>V</parameter>...</function>. Parameter values lose
    at most one leading and one trailing newline. Returns None if the body is malformed.
    """
    FO = "<function="
    PO = "<parameter="
    PC = "</parameter>"
    fi = body.find(FO)
    if fi == -1:
        return None
    fi += len(FO)
    fe = body.find(">", fi)
    if fe == -1:
        return None
    name = body[fi:fe]
    if not name:
        return None
    c = ToolCall(name=name)
    pos = fe + 1
    while True:
        p = body.find(PO, pos)
        if p == -1:
            break
        ke = body.find(">", p + len(PO))
        if ke == -1:
            return None
        ve = body.find(PC, ke + 1)
        if ve == -1:
            return None
        key = body[p + len(PO):ke]
        val = _strip_value(body[ke + 1:ve])
        c.args.append((key, val))
        pos = ve + len(PC)
    return c
