import re
import subprocess
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence, Union, Any

from FairFederatedDataset import FairFederatedDataset


Replacement = Union[str, Sequence[str]]
SOURCE_FILE = Path("datasheet_template.md")

def get_git_info(repo: Path | str = ".",
                 remote_name: str = "origin"):
    """
    Return the current commit SHA and the fetch URL of *remote_name*
    for the repository rooted at *repo* (default: current directory).

    Raises RuntimeError if *repo* isn't inside a Git repo or if Git isn't on PATH.
    """
    repo = Path(repo).expanduser().resolve()

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repo,
                stderr=subprocess.STDOUT,
                text=True
            ).strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"git {' '.join(args)} failed: {e.output.strip()}.\n\
                                 Perhaps your remote is not called 'origin'?") from e

    commit_sha = run("rev-parse", "HEAD")

    # get remote URL (may not exist)
    try:
        remote_url = run("remote", "get-url", remote_name)
    except RuntimeError:
        remote_url = None

    return commit_sha, remote_url


def prep_info_dict(debug=False):
    tag_block = re.compile(
        r'\[tag:([^\]]+)\]'      
        r'(.*?)'                 
        r'\[/tag\]',             
        flags=re.DOTALL       
    )

    tags: dict[str, list[str]] = defaultdict(list)

    text = SOURCE_FILE.read_text(encoding="utf‑8")
    for m in tag_block.finditer(text):
        name, body = m.group(1), m.group(2).strip()
        tags[name].append(body)
    
    if debug:
        for name, bodies in tags.items():
            print(f"[{name}] → {len(bodies)} occurrence(s):")
            for i, payload in enumerate(bodies, 1):
                # `payload` might be '', empty
                if payload:
                    preview = payload.splitlines()[0][:60]
                    ellipsis = '…' if len(payload) > len(preview) else ''
                else:
                    preview, ellipsis = '(empty)', ''
                print(f"  {i:>2}. {preview}{ellipsis}")
    
    commit, remote = get_git_info()
    tags['commit'] = commit
    tags['remote'] = remote
    return tags


def create_new_datasheet(source: Path | str,
                         destination: Path | str,
                         dataset: FairFederatedDataset,
                         keep_missing: bool = True) -> None:
    """
    Copies source to destination, replacing every [tag:name]...[/tag] block
    with data about the input *dataset*.

    If a tag appears more times than you supplied
    values for, behaviour depends on *keep_missing*:
        * True  – leave a warning in the datasheet
        * False – raise a KeyError / IndexError
    """
    # prepare replacement dictionary and template regex
    replacements = prep_info_dict()
    TAG_BLOCK = re.compile(
      r'\[tag:([^\]]+)\].*?\[/tag\]',     
      flags=re.DOTALL
    )
    # create datasheet from template
    source = Path(source).expanduser().resolve()
    dest = Path(destination).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # parse dataset object representation
    data_json = json.loads(dataset.to_json())
    replacements['name'] = data_json['_dataset_name']
    replacements['year'] = data_json['_year'] + ' with horizon ' + data_json['_horizon']
    replacements['sensitivedescriptions'] = data_json['_sensitive_attribute']

    # track how many times we've used each tag in the template
    seen: dict[str, int] = defaultdict(int)

    def _replace(match: re.Match[str]) -> str:
        tag = match.group(1)
        idx = seen[tag]
        seen[tag] += 1
        print(replacements[tag])
        if replacements[tag] == "":
            if keep_missing:
                return "**To be Filled -- Incomplete Datasheet!**"
            raise KeyError(f"no replacement supplied for tag '{tag}'")

        value = replacements[tag]

        if isinstance(value, (list, tuple)):
            if idx >= len(value):
                if keep_missing:
                    return match.group(0)
                raise IndexError(f"tag '{tag}' seen {idx+1} times "
                                 f"but only {len(value)} replacement(s) given")
            return str(value[idx])
        else:
            return str(value)

    dest.write_text(TAG_BLOCK.sub(_replace, source.read_text(encoding="utf‑8")),
                    encoding="utf‑8")

if __name__ == '__main__':
    dataset = FairFederatedDataset(
                    dataset="ACSIncome",
                    states=["CT", "DE"],
                    partitioners={"CT": 2, "DE": 1},
                    train_test_split=None,
                    fairness_metric='EO',
                    individual_fairness='attribute',
    )
    create_new_datasheet(SOURCE_FILE, 'sheet_test.md', dataset)
