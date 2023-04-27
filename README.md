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

> *Why not `conda`?*
>
> `conda` is great because it can manage the many non-Python dependencies modern
> machine learning depends on. But I couldn't get it to work well with editable
> installs or combine `conda` environments easily. Ideally, I'd specify the dependencies
> of each project with `conda-build`, but that was a lot of work. Manually setting
> up a separate `environment.yml` file per project was fine, but I couldn't import
> other local projects while getting their dependencies.

## PaperSpace

Installing NVIDIA drivers was annoying. I used the GCP script to install it, but
I ran into problems because the Nouveau driver wasn't getting removed automatically.
So I created a blacklist for it by following [these instructions](https://gist.github.com/wangruohui/df039f0dc434d6486f5d4d098aa52d07#creat-blacklist-for-nouveau-driver).

## GCP setup

GCP sucks. When I started up my VM instance a couple days later, I got resource
exhaustion. There's no transparency on which regions have available resources.

I'm trying PaperSpace which is more transparent. Let's see if I run into issues
here too.

### Create instance

Choose some compute zone. You can see all available ones with:

```bash
gcloud compute zones list
```

Find which machine types are available in the desired zone:

```bash
gcloud compute machine-types list --filter="zone:( europe-west1-b )"
```

I don't like using the Linux for Deep Learning boot images because they have some
dependencies installed via `conda`. That caused some problems with `pyenv`, and
it was simpler to just not have to worry about `conda`. I installed the base Buster
(Debian 10) image. Then:

- [Install CUDA drivers](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#no-secure-boot)
- Install `git`: `sudo apt install git-all`
- Install `pyenv` after [installing dependencies](https://github.com/pyenv/pyenv/wiki)

### Setup SSH and remote desktop with VSCode

It's nicer to add SSH keys to your GCP project metadata so that you can SSH into
any instance without extra setup. [Here's a guide](https://cloud.google.com/compute/docs/connect/add-ssh-keys#add_ssh_keys_to_project_metadata).

> You might need to go to "Metadata" on GCP and re-add the SSH key if you add a
> new instance.

Now just SSH. For example, the username and key I use is:

```bash
ssh -i ~/.ssh/gcp wba@EXTERNAL_IP
```

It's also nice to add this config to your `~/.ssh/config` with some alias so you
can just `ssh t4`.
