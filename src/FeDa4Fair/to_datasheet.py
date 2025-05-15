import re
import subprocess
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence, Union, Any
from datetime import datetime

import pandas as pd
from FairFederatedDataset import FairFederatedDataset


Replacement = Union[str, Sequence[str]]
SOURCE_FILE = Path("datasheet_template.md")


def get_git_info(repo: Path | str = ".", remote_name: str = "origin"):
    repo = Path(repo).expanduser().resolve()

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo, stderr=subprocess.STDOUT, text=True
            ).strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"git {' '.join(args)} failed: {e.output.strip()}.\n"
                "Perhaps your remote is not called 'origin'?"
            ) from e

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
    for split_name, split_ds in ffd._dataset.items():
        df = pd.DataFrame(split_ds)
        split_total = len(df)

        split_props[split_name] = {}
        for sa in sens:
            vc = (
                df[sa].value_counts(normalize=True)
                .round(decimal_places)
                .to_dict()
            )
            split_props[split_name][sa] = vc
            overall_counts[sa].update(df[sa])

        overall_total += split_total

    # 2) Per-partition stats
    for split_name, partitioner in ffd._partitioners.items():
        part_props[split_name] = {}
        for pid in range(partitioner.num_partitions):
            pdf = pd.DataFrame(partitioner.load_partition(partition_id=pid))
            part_props[split_name][pid] = {
                sa: pdf[sa]
                .value_counts(normalize=True)
                .round(decimal_places)
                .to_dict()
                for sa in sens
            }

    overall_props = {
        sa: {
            k: round(v / overall_total, decimal_places)
            for k, v in cnt.items()
        }
        for sa, cnt in overall_counts.items()
    }

    return {
        "overall": overall_props,
        "splits": split_props,
        "partitions": part_props,
    }



def prep_info_dict(debug: bool = False):
    tag_block = re.compile(r"\[tag:([^\]]+)\](.*?)\[/tag\]", flags=re.DOTALL)

    tags: dict[str, list[str]] = defaultdict(list)

    text = SOURCE_FILE.read_text(encoding="utf-8")
    for m in tag_block.finditer(text):
        name, body = m.group(1), m.group(2).strip()
        tags[name].append(body)

    if debug:
        for name, bodies in tags.items():
            print(f"[{name}] → {len(bodies)} occurrence(s):")
            for i, payload in enumerate(bodies, 1):
                if payload:
                    preview = payload.splitlines()[0][:60]
                    ellipsis = "…" if len(payload) > len(preview) else ""
                else:
                    preview, ellipsis = "(empty)", ""
                print(f"  {i:>2}. {preview}{ellipsis}")

    commit, remote = get_git_info()
    tags["commit"] = commit
    tags["remote"] = remote
    return tags


# ----------------------------------------------------------------------
# 2. Build the datasheet
# ----------------------------------------------------------------------
def create_new_datasheet(
    destination: Path | str,
    dataset: FairFederatedDataset,
    keep_missing: bool = True,
) -> None:
    KEEP = "KEEP"
    DROP = None 

    replacements: dict[str, Any] = prep_info_dict()

    source = Path(SOURCE_FILE).expanduser().resolve()
    dest = Path(destination).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    data_json = json.loads(dataset.to_json())

    # ACSIncome or ACSEmployment?
    replacements["income"] = KEEP if data_json["_dataset_name"] == "ACSIncome" else DROP
    replacements["employment"] = KEEP if data_json["_dataset_name"] == "ACSEmployment" else DROP

    # basic fields
    replacements["name"] = data_json["_dataset_name"] + 'FeDa4Fair' + datetime.now().strftime("%Y-%m-%d")
    replacements["year"] = f"{data_json['_year']} with horizon {data_json['_horizon']}"
    replacements["sensitivedescriptions"] = data_json["_sensitive_attributes"]

    # unit of analysis – individuals vs. households
    replacements["individuals"] = (
        "individual" if data_json["_dataset_name"] == "ACSEmployment" else "household"
    )

    replacements["sens_remaining"] = (
        "Yes" if dataset._sensitive_attributes else "No"
    )

    # sensitive attribute proportions
    sens_stats = compute_sensitive_attr_proportions(dataset)
    replacements["sens_overall"] = json.dumps(sens_stats["overall"], indent=2)
    replacements["sens_by_split"] = json.dumps(sens_stats["splits"], indent=2)
    replacements["sens_by_partition"] = json.dumps(
        sens_stats["partitions"], indent=2
    )

    # modification information
    repl = dataset._modification_dict
    if repl is not None:
        replacements["modification"] = json.dumps(repl, indent=2)
    else:
        replacements["modification"] = 'No modification was done.'

    # column mames
    colnames = next(iter(dataset._dataset.values())).column_names[:-1]
    replacements["columns"] = json.dumps(colnames)

    # number of rows
    nrows = sum(len(split) for split in dataset._dataset.values())
    replacements["nrows"] = json.dumps(nrows)

    TAG_BLOCK = re.compile(
        r"\[tag:([^\]]+)\](.*?)\[/tag\]",   # ← (.*?) is now group 2
        flags=re.DOTALL
    )
    seen: dict[str, int] = defaultdict(int)

    def _replace(match: re.Match[str]) -> str:
        tag  = match.group(1)
        body = match.group(2)
        idx  = seen[tag]; seen[tag] += 1

        if tag not in replacements:
            return "**To be Filled -- Incomplete Datasheet!**" if keep_missing else match.group(0)

        value = replacements[tag]

        if value is DROP:
            return ""
        if value is KEEP:
            return body.strip()

        if isinstance(value, (list, tuple)):
            if idx >= len(value):
                return "**To be Filled -- Incomplete Datasheet!**" if keep_missing else match.group(0)
            return str(value[idx])

        if value == "":
            return "**To be Filled -- Incomplete Datasheet!**" if keep_missing else ""

        return str(value)    

    dest.write_text(
        TAG_BLOCK.sub(_replace, source.read_text(encoding="utf-8")),
        encoding="utf-8",
    )



if __name__ == "__main__":
    dataset = FairFederatedDataset(
        dataset="ACSIncome",
        states=["CT", "DE"],
        partitioners={"CT": 2, "DE": 1},
        fairness_metric="DP",
        fairness_level="attribute",
        modification_dict={"CT": {
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
        }
    )
    
    partition_CT_0 = dataset.load_partition(split="CT", partition_id=0)
    split_CT = dataset.load_split("CT")
    dataset.save_dataset("data_fl")
    create_new_datasheet("data_fl/datasheet.md", dataset)
