A runtime typechecker, mostly to use with `jaxtyping`.

`beartype` randomly checks pieces of arguments and is too big and complex. `typeguard` doesn't work for Jupyter notebooks because it fetches the AST. I don't need AST parsing because I just want to typecheck the function signature via a decorator.
