"""Build uploadable data archives from an explicit file list on the data server.

No study data are downloaded by this command. Every archive contains paths
relative to the reader's working directory, plus a file list and instructions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def safe_name(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Archive path must be relative: {value}")
    return str(path)


def build_archive(bundle: dict, output: Path, reserve_bytes: int) -> dict:
    name = safe_name(bundle["filename"])
    if "/" in name or not name.endswith(".zip"):
        raise ValueError("Use a ZIP filename without a directory")
    target = output / name
    files = bundle["files"]
    seen = set()
    listing = []
    for item in files:
        source = Path(item["source"])
        relative = safe_name(item["destination"])
        if relative in seen:
            raise ValueError(f"Duplicate archive entry: {relative}")
        seen.add(relative)
        if not source.is_file():
            raise FileNotFoundError(source)
        listing.append({"path": relative, "bytes": source.stat().st_size})
    if target.exists():
        raise FileExistsError(f"Choose a new output directory: {target}")
    partial = target.with_suffix(".zip.part")
    if partial.exists():
        raise FileExistsError(partial)
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True) as archive:
        for item, record in zip(files, listing):
            print(f"{name}: {record['path']}", flush=True)
            with Path(item["source"]).open("rb") as source:
                info = zipfile.ZipInfo(record["path"])
                info.compress_type = zipfile.ZIP_DEFLATED
                with archive.open(info, "w", force_zip64=True) as destination:
                    while chunk := source.read(8 * 1024 * 1024):
                        if shutil.disk_usage(output).free < reserve_bytes:
                            raise OSError("Stopped before filling the data disk. Keep the .part file for inspection.")
                        destination.write(chunk)
        archive.writestr("README_" + name.removesuffix(".zip") + ".txt", bundle["instructions"])
        archive.writestr("FILES_" + name.removesuffix(".zip") + ".json", json.dumps(listing, indent=2))
    with zipfile.ZipFile(partial) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"ZIP integrity check failed: {bad}")
    partial.rename(target)
    return {"filename": name, "dataset": bundle["dataset"], "purpose": bundle["purpose"],
            "archive_bytes": target.stat().st_size, "uncompressed_bytes": sum(f["bytes"] for f in listing),
            "file_count": len(listing), "status": "ready_to_upload"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--reserve-gb", type=float, default=3)
    args = parser.parse_args()
    bundles = json.loads(args.plan.read_text())["bundles"]
    if args.dataset:
        bundles = [b for b in bundles if b["dataset"] in args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / ("downloads_" + "_".join(args.dataset or ["all"]) + ".json")
    if report.exists():
        raise FileExistsError(report)
    results = []
    for bundle in bundles:
        results.append(build_archive(bundle, args.output_dir, int(args.reserve_gb * 1e9)))
        report.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results[-1]), flush=True)


if __name__ == "__main__":
    main()
