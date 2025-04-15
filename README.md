# fairFL-data


## Create the environment

First of all we need to install [uv](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then we can create the environment:

```bash
uv sync
uv venv
```

## Run Formatting 

```bash
uv run ruff format
```
