import functools
import inspect
import typing


def typechecker(f):
    sig = inspect.signature(f)
    hints = typing.get_type_hints(f, include_extras=True)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Check parameters
        for name, val in bound.arguments.items():
            if name not in hints:
                continue
            hint = hints[name]
            if hint is not typing.Any and not isinstance(val, hint):
                raise TypeError(f"{f.__name__}() param '{name}' expected {hint}, got {val}")

        out = f(*args, **kwargs)

        # Check return value
        if "return" in hints:
            hint = hints["return"]
            if hint is not typing.Any and not isinstance(out, hint):
                raise TypeError(f"{f.__name__}() return expected {hint}, got {out}")

        return out
    return wrapper

__all__ = ["typechecker"]
