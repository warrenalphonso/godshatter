from jaxtyping import install_import_hook

with install_import_hook("nanogpt", "typechecker.typechecker"):
    import nanogpt.model
