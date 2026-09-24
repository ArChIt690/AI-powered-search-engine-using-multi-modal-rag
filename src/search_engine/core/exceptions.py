class UnsupportedFileTypeError(ValueError):
    """The file's extension has no loader in Ingestion (e.g. .exe, .docx)."""
