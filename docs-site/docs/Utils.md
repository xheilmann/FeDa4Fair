---
sidebar_position: 6
---

# Utilities

General utility functions for data manipulation and bias injection.

## Functions

### `drop_data`

```python
def drop_data(
    df: pd.DataFrame,
    percentage: float,
    column1: str,
    value1: Any,
    label_column: str,
    column2: str | None = None,
    value2: Any | None = None,
) -> pd.DataFrame:
```

Drop a percentage of rows from a DataFrame that match specific criteria.

**Parameters:**
- `df`: Input DataFrame.
- `percentage`: Fraction of matching rows to drop (0.0 - 1.0).
- `column1`: First filter column.
- `value1`: Value to match in `column1`.
- `label_column`: Label column name (must be True/1 for row to be dropped).
- `column2`: Optional second filter column.
- `value2`: Optional value to match in `column2`.

---

### `flip_data`

```python
def flip_data(
    df: pd.DataFrame,
    percentage: float,
    column1: str,
    value1: Any,
    label_column: str,
    column2: str | None = None,
    value2: Any | None = None,
) -> pd.DataFrame:
```

Flip the label from True to False for a percentage of rows matching specified criteria.

**Parameters:**
- `df`: Input DataFrame.
- `percentage`: Fraction of matching rows to flip (0.0 - 1.0).
- `column1`: First filter column.
- `value1`: Value to match in `column1`.
- `label_column`: Label column name (label is flipped from True to False).
- `column2`: Optional second filter column.
- `value2`: Optional value to match in `column2`.
