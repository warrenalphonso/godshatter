import jaxtyping
from tinygrad.dtype import dtypes

basename = lambda dtype: str(dtype).split(".")[-1]

class Float(jaxtyping.AbstractDtype):
    dtypes = set(basename(dtype) for dtype in dtypes.floats)

class Int(jaxtyping.AbstractDtype):
    dtypes = set(basename(dtype) for dtype in dtypes.ints + dtypes.uints)

class Bool(jaxtyping.AbstractDtype):
    dtypes = [basename(dtypes.bool)]
