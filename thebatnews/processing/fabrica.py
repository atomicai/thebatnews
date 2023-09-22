def pure(x: str, **kwargs):
    return x.strip()


def register(x: str, cased: bool = False, **kwargs):
    if cased:
        return x
    return x.lower()


def running_ie5(x: str, prefix: str):
    x = pure(x)
    x = register(x)
    return prefix + x


def running(x: str):
    x = pure(x)
    x = register(x)
    return x


__all__ = ["running_ie5", "running"]
