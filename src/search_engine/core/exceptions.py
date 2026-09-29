class UnsupportedFileTypeError(ValueError):
    """The file's extension has no loader in Ingestion (e.g. .exe, .docx)."""


class LLMUnavailableError(RuntimeError):
    """The LLM call failed (outage, rate limit, timeout); the question can't be answered right now."""
