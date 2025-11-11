import typing

import pytest

from typechecker import check_type


def test_any():
    assert check_type(4, typing.Any)

@pytest.mark.parametrize("val,hint", [
    ((4, 'a'), tuple),
    ((4, 'a'), tuple[int, str]),
    ((4, 4, 4), tuple[int, ...]),
    ((), tuple[()]),
    (((1,),), tuple[tuple[int]])
])
def test_tuple(val, hint):
    assert check_type(val, hint)
