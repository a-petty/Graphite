"""Graphited daemon wire protocol.

Newline-delimited JSON, one request or response per line, over a Unix
domain socket at ``~/.graphite/daemon.sock``. The protocol shape is
deliberately JSON-RPC-ish so call-sites read naturally, but we keep only
the pieces we need — no batching, no notifications, no protocol version
negotiation.

Request:   {"id": <int>, "method": <str>, "params": <object>}
Response:  {"id": <int>, "result": <any>}
           {"id": <int>, "error": {"code": <int>, "message": <str>, "details": <any?>}}

All messages MUST fit on a single line (no embedded newlines). Long
responses stream as one long line; the client reads until the delimiter.

The `params` object may be missing or empty. `id` is chosen by the client
and echoed in the response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional


class ErrorCode(IntEnum):
    """Subset of JSON-RPC 2.0 error codes plus Graphite-specific ones.

    We use negative numbers to avoid colliding with anything application code
    might want to emit as a positive result.
    """

    # Standard JSON-RPC style
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # Graphite-specific
    NOT_READY = -32001       # graph not initialized yet (e.g., cold daemon)
    NOT_IMPLEMENTED = -32002 # stub method (e.g., ingest_source in Phase 1)
    BACKPRESSURE = -32003    # e.g., spool full, retry later


class ProtocolError(Exception):
    """Raised when a wire message cannot be parsed or is structurally bad."""

    def __init__(self, code: ErrorCode, message: str, details: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass
class Request:
    id: int
    method: str
    params: dict = field(default_factory=dict)

    def to_line(self) -> bytes:
        """Serialize to a newline-terminated UTF-8 line."""
        obj = {"id": self.id, "method": self.method, "params": self.params}
        return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")

    @classmethod
    def from_line(cls, line: bytes) -> "Request":
        try:
            obj = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ProtocolError(ErrorCode.PARSE_ERROR, f"Invalid JSON: {e}")
        if not isinstance(obj, dict):
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Request must be a JSON object")
        rid = obj.get("id")
        method = obj.get("method")
        params = obj.get("params", {})
        if not isinstance(rid, int):
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Request.id must be an integer")
        if not isinstance(method, str) or not method:
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Request.method must be a non-empty string")
        if not isinstance(params, dict):
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Request.params must be an object")
        return cls(id=rid, method=method, params=params)


@dataclass
class Response:
    id: int
    result: Any = None
    error: Optional[dict] = None  # {"code", "message", "details"?}

    def to_line(self) -> bytes:
        obj: dict = {"id": self.id}
        if self.error is not None:
            obj["error"] = self.error
        else:
            obj["result"] = self.result
        return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")

    @classmethod
    def from_line(cls, line: bytes) -> "Response":
        try:
            obj = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ProtocolError(ErrorCode.PARSE_ERROR, f"Invalid JSON: {e}")
        if not isinstance(obj, dict):
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Response must be a JSON object")
        rid = obj.get("id")
        if not isinstance(rid, int):
            raise ProtocolError(ErrorCode.INVALID_REQUEST, "Response.id must be an integer")
        if "error" in obj:
            err = obj["error"]
            if not isinstance(err, dict) or "code" not in err or "message" not in err:
                raise ProtocolError(
                    ErrorCode.INVALID_REQUEST,
                    "Response.error must be an object with code and message",
                )
            return cls(id=rid, error=err)
        return cls(id=rid, result=obj.get("result"))


def make_error(code: ErrorCode, message: str, details: Any = None) -> dict:
    """Build the ``error`` payload for a Response."""
    err: dict = {"code": int(code), "message": message}
    if details is not None:
        err["details"] = details
    return err
