---
sidebar_position: 7
---

# Datasheet Generation

Tools for automatically generating datasheets (documentation) for your federated datasets.

## Functions

### `create_new_datasheet`

```python
def create_new_datasheet(
    destination: Path | str,
    dataset: FairFederatedDataset,
    keep_missing: bool = True,
) -> None:
```

Generate a filled datasheet markdown file from a template and dataset metadata.

**Parameters:**
- `destination` (*Path | str*): Output path for the datasheet.
- `dataset` (*FairFederatedDataset*): The dataset object containing metadata.
- `keep_missing` (*bool*): If True, retains placeholders for missing information.

---

### `compute_sensitive_attr_proportions`

```python
def compute_sensitive_attr_proportions(
    ffd: FairFederatedDataset,
    sensitive_attrs: Sequence[str] | None = None,
    decimal_places: int = 3,
) -> dict[str, Any]:
```

Calculates overall, per-split, and per-partition proportions of sensitive attributes. Useful for populating datasheet statistics.

---

### `get_git_info`

```python
def get_git_info(repo: Path | str = ".", remote_name: str = "origin") -> tuple[str, str | None]:
```

Retrieves the current git commit SHA and remote URL to track the exact version of the code used to generate the dataset.
