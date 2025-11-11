import functools
import inspect
import typing


def check_type(val, hint) -> bool:
    if hint is typing.Any:
        return True

    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin is tuple:
        if not isinstance(val, tuple): return False
        if not args: return True
        if len(args) == 2 and args[-1] == Ellipsis:
            return all(check_type(x, args[0]) for x in val)
        if len(args) != len(val): return False
        return all(check_type(x, t) for x, t in zip(val, args))

    return isinstance(val, hint)


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
            if not check_type(val, hint):
                raise TypeError(f"{f.__name__}() param '{name}' expected {hint}, got {val}")

        out = f(*args, **kwargs)

        # Check return value
        if "return" in hints:
            hint = hints["return"]
            if not check_type(out, hint):
                raise TypeError(f"{f.__name__}() return expected {hint}, got {out}")

        return out
    return wrapper
