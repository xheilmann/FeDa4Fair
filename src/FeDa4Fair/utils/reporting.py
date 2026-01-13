import json
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from FeDa4Fair.dataset.fair_dataset import FairFederatedDataset

Replacement = str | Sequence[str]
SOURCE_FILE = Path(__file__).parent.parent / "templates" / "datasheet_template.md"


def get_git_info(repo: Path | str = ".", remote_name: str = "origin") -> tuple[str, str | None]:
    """
    Retrieve current git commit SHA and remote URL.

    Parameters
    ----------
    repo : Path | str, default="."
        Path to the git repository.
    remote_name : str, default="origin"
        Name of the remote to get the URL for.

    Returns
    -------
    tuple[str, str | None]
        A tuple containing (commit_sha, remote_url).

    """
    repo = Path(repo).expanduser().resolve()

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["/usr/bin/git", *args], cwd=repo, stderr=subprocess.STDOUT, text=True
            ).strip()
        except subprocess.CalledProcessError as e:
            err_msg = f"git {' '.join(args)} failed: {e.output.strip()}.\nPerhaps your remote is not called 'origin'?"
            raise RuntimeError(err_msg) from e

    commit_sha = run("rev-parse", "HEAD")
    try:
        remote_url = run("remote", "get-url", remote_name)
    except RuntimeError:
        remote_url = None
    return commit_sha, remote_url


def compute_sensitive_attr_proportions(
    ffd: "FairFederatedDataset",
    sensitive_attrs: Sequence[str] | None = None,
    decimal_places: int = 3,
) -> dict[str, Any]:
    """Return overall / per-split / per-partition proportions of each sensitive attribute."""
    # Prepare the dataset if it hasn't been done yet
    if not getattr(ffd, "_dataset_prepared", False):
        ffd._prepare_dataset()

    sens = sensitive_attrs or ffd._sensitive_attributes or ["SEX", "MAR", "RAC1P"]

    overall_counts: Mapping[str, Counter] = {sa: Counter() for sa in sens}
    overall_total = 0

    split_props: dict[str, dict[str, Mapping[Any, float]]] = {}
    part_props: dict[str, dict[int, dict[str, Mapping[Any, float]]]] = {}

    # 1) Per-split stats
    if ffd._dataset is not None:
        for split_name, split_ds in ffd._dataset.items():
            df = pd.DataFrame(split_ds)
            split_total = len(df)

            split_props[split_name] = {}
            for sa in sens:
                vc = df[sa].value_counts(normalize=True).round(decimal_places).to_dict()
                split_props[split_name][sa] = vc
                overall_counts[sa].update(df[sa])

            overall_total += split_total

    # 2) Per-partition stats
    for split_name, partitioner in ffd._partitioners.items():
        part_props[split_name] = {}
        for pid in range(partitioner.num_partitions):
            pdf = pd.DataFrame(partitioner.load_partition(partition_id=pid))
            part_props[split_name][pid] = {
                sa: pdf[sa].value_counts(normalize=True).round(decimal_places).to_dict() for sa in sens
            }

    overall_props = {
        sa: {k: round(v / overall_total, decimal_places) for k, v in cnt.items()} for sa, cnt in overall_counts.items()
    }

    return {
        "overall": overall_props,
        "splits": split_props,
        "partitions": part_props,
    }


def prep_info_dict(debug: bool = False) -> dict[str, Any]:
    """
    Parse the datasheet template to identify tags and collect git info.
    """
    tags = _parse_tags()
    if debug:
        _print_debug_tags(tags)

    commit, remote = get_git_info()
    tags["commit"] = [commit]
    if remote:
        tags["remote"] = [remote]
    else:
        tags["remote"] = []
    return tags


def _parse_tags() -> dict[str, list[str]]:
    """Extract tags from the template file."""
    tag_block = re.compile(r"\[tag:([^\]]+)\](.*?)\[/tag\]", flags=re.DOTALL)
    tags: dict[str, list[str]] = defaultdict(list)
    text = SOURCE_FILE.read_text(encoding="utf-8")
    for m in tag_block.finditer(text):
        tags[m.group(1)].append(m.group(2).strip())
    return tags


def _print_debug_tags(tags: dict[str, list[str]]) -> None:
    """Print found tags for debugging."""
    for name, bodies in tags.items():
        print(f"[{name}] → {len(bodies)} occurrence(s):")
        for i, payload in enumerate(bodies, 1):
            preview = payload.splitlines()[0][:60] if payload else "(empty)"
            ellipsis = "…" if len(payload) > len(preview) else ""
            print(f"  {i:>2}. {preview}{ellipsis}")


# ----------------------------------------------------------------------
# 2. Build the datasheet
# ----------------------------------------------------------------------
def create_new_datasheet(
    destination: Path | str,
    dataset: FairFederatedDataset,
    keep_missing: bool = True,
) -> None:
    """
    Generate a filled datasheet markdown file from the template and dataset metadata.
    """
    replacements = _get_datasheet_replacements(dataset)
    dest = Path(destination).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    TAG_BLOCK = re.compile(r"\[tag:([^\]]+)\](.*?)\[/tag\]", flags=re.DOTALL)
    seen: dict[str, int] = defaultdict(int)

    def _replace(match: re.Match[str]) -> str:
        tag, body = match.group(1), match.group(2)
        idx = seen[tag]
        seen[tag] += 1
        return _get_tag_replacement(tag, body, idx, replacements, keep_missing, match.group(0))

    dest.write_text(TAG_BLOCK.sub(_replace, SOURCE_FILE.read_text(encoding="utf-8")), encoding="utf-8")


def _get_tag_replacement(tag, body, idx, replacements, keep_missing, original):
    """Determine the replacement text for a single tag."""
    if tag not in replacements:
        return "**To be Filled -- Incomplete Datasheet!**" if keep_missing else original

    value = replacements[tag]
    if value is None:
        return ""  # DROP
    if value == "KEEP":
        return body.strip()

    if isinstance(value, (list, tuple)):
        if idx >= len(value):
            return "**To be Filled -- Incomplete Datasheet!**" if keep_missing else original
        return str(value[idx])

    return str(value) if value != "" else ("**To be Filled -- Incomplete Datasheet!**" if keep_missing else "")


def _get_datasheet_replacements(dataset: FairFederatedDataset) -> dict[str, Any]:
    """Gather all replacement values for the datasheet tags."""
    replacements = prep_info_dict()
    data_json = json.loads(dataset.to_json())

    replacements.update(_get_dataset_basic_info(data_json, dataset))
    replacements.update(_get_dataset_stats_info(dataset))

    repl = dataset._modification_dict
    replacements["modification"] = json.dumps(repl, indent=2) if repl is not None else "No modification was done."
    return replacements


def _get_dataset_basic_info(data_json: dict, dataset: FairFederatedDataset) -> dict[str, Any]:
    """Extract basic descriptive information from the dataset."""
    is_income = data_json["_dataset_name"] == "ACSIncome"
    is_employment = data_json["_dataset_name"] == "ACSEmployment"

    return {
        "income": "KEEP" if is_income else None,
        "employment": "KEEP" if is_employment else None,
        "name": data_json["_dataset_name"] + "FeDa4Fair" + datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "year": f"{data_json['_year']} with horizon {data_json['_horizon']}",
        "sensitivedescriptions": data_json["_sensitive_attributes"],
        "individuals": "individual" if is_employment else "household",
        "sens_remaining": "Yes" if dataset._sensitive_attributes else "No",
    }


def _get_dataset_stats_info(dataset: FairFederatedDataset) -> dict[str, Any]:
    """Compute statistics and proportions for the datasheet."""
    sens_stats = compute_sensitive_attr_proportions(dataset)

    if dataset._dataset is None:
        colnames, nrows = [], 0
    else:
        colnames = next(iter(dataset._dataset.values())).column_names[:-1]
        nrows = sum(len(split) for split in dataset._dataset.values())

    return {
        "sens_overall": json.dumps(sens_stats["overall"], indent=2),
        "sens_by_split": json.dumps(sens_stats["splits"], indent=2),
        "sens_by_partition": json.dumps(sens_stats["partitions"], indent=2),
        "columns": json.dumps(colnames),
        "nrows": json.dumps(nrows),
    }


if __name__ == "__main__":
    dataset = FairFederatedDataset(
        dataset="ACSIncome",
        states=["CT", "DE"],
        partitioners={"CT": 2, "DE": 1},
        fairness_metric="DP",
        fairness_level="attribute",
        modification_dict={
            "CT": {
                "MAR": {
                    "drop_rate": 0.6,
                    "flip_rate": 0.3,
                    "value": 1,
                    "attribute": "SEX",
                    "attribute_value": 1,
                },
                "SEX": {
                    "drop_rate": 0.5,
                    "flip_rate": 0.2,
                    "value": 2,
                    "attribute": None,
                    "attribute_value": None,
                },
            }
        },
    )

    partition_CT_0 = dataset.load_partition(split="CT", partition_id=0)
    split_CT = dataset.load_split("CT")
    dataset.save_dataset(Path("data_fl"))
    create_new_datasheet("data_fl/datasheet.md", dataset)
