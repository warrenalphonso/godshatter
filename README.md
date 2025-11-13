# godshatter

> Being a thousand independent shards of ideas isn't always fun, but at least
> it's not *boring*.

## Development

Install and use [`pre-commit`](https://pre-commit.com/) git hooks. `pre-commit`
rules should be really generic and apply to the entire repo. If we want more specific
per-project rules, we'll need a better git hooks project.

## Dependency management

Use `uv sync` to install library dependencies. They might install editable
versions of other libraries in the repo.

## Compute

Use [SkyPilot](https://docs.skypilot.co/en/latest/docs/index.html) to provision
compute infra and run scripts. I think it's pretty useful to install globally
with `pipx install skypilot`.
