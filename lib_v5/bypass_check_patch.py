import inspect


_original_getsource = inspect.getsource


def _safe_getsource(obj):
    try:
        return _original_getsource(obj)
    except OSError:
        return ""


inspect.getsource = _safe_getsource
