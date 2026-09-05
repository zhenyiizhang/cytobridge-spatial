"""Download the paper data and trained models into a working directory."""
from __future__ import annotations

import argparse
import hashlib
from http.client import RemoteDisconnected
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from urllib.request import Request, urlopen
from urllib.error import URLError
import zipfile


_MANIFEST = Path(__file__).parent / "results/data/downloads/manifest.json"


def _open_download(part):
    headers = {'User-Agent': 'CytoBridge-data-download', 'Accept': 'application/octet-stream'}
    if part.get('bytes'):
        headers['Range'] = f"bytes=0-{part['bytes'] - 1}"
    try:
        return urlopen(Request(part['url'], headers=headers), timeout=20)
    except (URLError, TimeoutError, RemoteDisconnected, ConnectionError):
        if not part.get('asset_id'):
            raise
        # Some networks reach GitHub's API but not its main download hostname.
        # The public asset endpoint needs no account or access token.
        alternative = ('https://api.github.com/repos/zhenyiizhang/cytobridge-spatial/'
                       f"releases/assets/{part['asset_id']}")
        return urlopen(Request(alternative, headers=headers), timeout=120)


def _member_path(name: str, root: Path) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Invalid archive path: {name}")
    path = root.joinpath(*relative.parts).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Archive path leaves the output directory: {name}")
    return path


def _extract(archive: Path, destination: Path) -> list[dict]:
    files = []
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for member in members:
            path = _member_path(member.filename, destination)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive contains a symbolic link: {member.filename}")
            if not member.is_dir() and path.exists():
                raise FileExistsError(f"Use an empty output directory or move this file first: {path}")
        for member in members:
            path = _member_path(member.filename, destination)
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            files.append({"path": member.filename, "bytes": member.file_size})
    return files


def _download_archive(record: dict, destination: Path) -> None:
    receipt = destination / ".cytobridge" / (record["archive"] + ".json")
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get("sha256") == record["sha256"] and all(
            _member_path(f["path"], destination).is_file()
            and _member_path(f["path"], destination).stat().st_size == f["bytes"]
            for f in previous["files"]
        ):
            print(f"Already downloaded: {record['archive']}")
            return
        raise FileExistsError(f"Existing download differs. Choose a new directory: {destination}")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    # Numbered parts are joined while downloading, so only one ZIP is stored.
    with tempfile.TemporaryDirectory(prefix="download-", dir=receipt.parent) as temporary:
        archive = Path(temporary) / record["archive"]
        total_digest = hashlib.sha256()
        total_bytes = 0
        with archive.open("xb") as handle:
            for part in record["parts"]:
                print(f"Downloading {part['name']} ({part['bytes'] / 1e6:.0f} MB)", flush=True)
                digest = hashlib.sha256()
                received = 0
                with _open_download(part) as response:
                    while block := response.read(8 * 1024 * 1024):
                        handle.write(block)
                        digest.update(block)
                        total_digest.update(block)
                        received += len(block)
                if received != part["bytes"] or digest.hexdigest() != part["sha256"]:
                    raise OSError(f"Incomplete download: {part['name']}. Run the command again.")
                total_bytes += received
        if total_bytes != record["bytes"] or total_digest.hexdigest() != record["sha256"]:
            raise OSError(f"The downloaded archive is incomplete: {record['archive']}")
        files = _extract(archive, destination)
    receipt.write_text(json.dumps({"sha256": record["sha256"], "files": files}, indent=2) + "\n")


def download(dataset: str, destination: str | Path = ".", *, kind: str = "analysis") -> Path:
    """Download and extract the files for one paper dataset.

    Parameters
    ----------
    dataset
        Dataset name, for example ``chicken_heart`` or ``mosta``.
    destination
        Working directory. Archives create ``data/<dataset>/`` below it.
    kind
        ``analysis`` downloads the model and aligned data used by the dataset
        tutorial. ``all`` also downloads saved paper populations and any raw
        training inputs distributed for that dataset. An archive name can
        select a single additional download.

    Returns
    -------
    pathlib.Path
        The dataset directory. Completed downloads are reused on later calls.
        Existing unrelated files are never overwritten.
    """
    manifest = json.loads(_MANIFEST.read_text())
    available = {item["archive"]: item for item in manifest["archives"]}
    names = [name for name in available if name.startswith(dataset + "_")]
    if not names:
        raise ValueError(f"Unknown dataset: {dataset}")
    if kind == "analysis":
        names = [f"{dataset}_model.zip", f"{dataset}_analysis_data.zip"]
    elif kind != "all":
        if kind not in names:
            raise ValueError(f"No {kind!r} archive for {dataset}")
        names = [kind]
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        _download_archive(available[name], destination)
    return destination / "data" / dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--kind", default="analysis", help="analysis, all, or an archive filename")
    args = parser.parse_args()
    print(download(args.dataset, args.output_dir, kind=args.kind))


if __name__ == "__main__":
    main()
