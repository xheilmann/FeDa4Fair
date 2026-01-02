---
sidebar_position: 5
---

# Partitioning

FeDa4Fair provides custom partitioning strategies to support fairness-aware federated learning scenarios.

## Classes

### `RepresentativeDiversityPartitioner`

```python
class RepresentativeDiversityPartitioner(Partitioner):
    def __init__(
        self,
        num_partitions: int,
        partition_by: str | list[str],
        seed: int | None = 42,
    ) -> None:
        ...
```

A partitioner that ensures representative diversity by stratifying the dataset based on one or more sensitive attributes. Each partition will receive an equal proportion of samples from each subgroup defined by the sensitive attributes.

**Parameters:**

- **`num_partitions`** (*int*): The total number of partitions to create.
- **`partition_by`** (*str | list[str]*): The column name(s) (sensitive attributes) to stratify by.
- **`seed`** (*Optional[int]*, default=42): Random seed for reproducibility.

**Usage Example:**

```python
from FeDa4Fair.dataset.partitioning import RepresentativeDiversityPartitioner

# Partition based on 'sex' and 'race' to ensure diversity in each client
partitioner = RepresentativeDiversityPartitioner(
    num_partitions=10,
    partition_by=["sex", "race"]
)
```
