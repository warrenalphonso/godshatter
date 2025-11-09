# godshatter

> Being a thousand independent shards of ideas isn't always fun, but at least
> it's not *boring*.

## Development

Install and use [`pre-commit`](https://pre-commit.com/) git hooks. `pre-commit`
rules should be really generic and apply to the entire repo. If we want more specific
per-project rules, we'll need a better git hooks project.

## Dependency management

The projects in `godshatter` aren't meant to be built or packaged, so the project
metadata is pretty simple!


1. We use `pip` and [requirements files](https://pip.pypa.io/en/stable/reference/requirements-file-format/#requirements-file-format)
   to specify dependencies. Requirements files are great because they allow us to
   reference other local projects and their dependencies. Plus, they're simple and
   well-maintained.
2. We use [`pyproject.toml`](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#declaring-project-metadata)
   to track basic project metadata.
