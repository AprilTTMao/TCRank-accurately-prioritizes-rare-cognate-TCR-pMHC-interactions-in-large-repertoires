#!/usr/bin/env python3
"""Rank candidate TCRs for peptide-HLA targets using selected predictors."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
IMAGE = "tcrank-cli:cpu-20261015"
ARCHIVE = "tcrank-cli-cpu-20261015.tar.gz"
AA = set("ACDEFGHIKLMNPQRSTVWY")


def mode_name(value):
    aliases = {"cdr": "cdr3", "cdr3": "cdr3", "v": "vregion", "vregion": "vregion"}
    try:
        return aliases[value.lower()]
    except KeyError:
        raise argparse.ArgumentTypeError("Choose cdr3 (cdr) or vregion (v).")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer.")
    return number


def method_config():
    return json.loads((ROOT / "core/tools_config.json").read_text())["methods"]


def select_methods(names, mode):
    configured = method_config()
    names = [part for name in names for part in name.split(",") if part]
    available = [item["name"] for item in configured if mode in item["allow_modes"]]
    if [name.lower() for name in names] == ["all"]:
        return available
    lookup = {item["name"].lower(): item["name"] for item in configured}
    selected = []
    for name in names:
        method = lookup.get(name.strip().lower())
        if method is None:
            raise ValueError("Unknown predictor: " + name + ". Use --list-predictors.")
        if method not in available:
            raise ValueError(method + " requires --mode vregion.")
        if method not in selected:
            selected.append(method)
    if not selected:
        raise ValueError("Select at least one predictor.")
    return selected


def read_input(path, mode):
    """Validate every row; never silently drop an invalid candidate."""
    aliases = {
        "alpha": ("alpha_cdr3", "alpha_v", "tcr_alpha", "alpha"),
        "beta": ("beta_cdr3", "beta_v", "tcr_beta", "beta"),
        "peptide": ("peptide",),
        "mhc": ("mhc", "hla", "mhc/hla", "mhc_hla"),
    }
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        normalized = [field.strip().lower() for field in fields]
        if len(set(normalized)) != len(normalized):
            raise ValueError("CSV contains duplicate column names.")
        columns = dict(zip(normalized, fields))
        chosen = {}
        for key, alternatives in aliases.items():
            matches = [columns[name] for name in alternatives if name in columns]
            if len(matches) > 1:
                raise ValueError("Ambiguous " + key + " columns: " + ", ".join(matches))
            chosen[key] = matches[0] if matches else None
        required = ["beta", "peptide", "mhc"] + (["alpha"] if mode == "vregion" else [])
        if any(chosen[key] is None for key in required):
            raise ValueError("CSV requires beta, peptide and mhc/HLA columns; V-region also requires alpha.")
        opposite = "_v" if mode == "cdr3" else "_cdr3"
        if any(chosen[key] and chosen[key].strip().lower().endswith(opposite) for key in ("alpha", "beta")):
            raise ValueError("Sequence column names do not match --mode.")
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed CSV at row " + str(number))
            if not any(value.strip() for value in row.values()):
                continue
            cleaned = {key: (row.get(column, "") if column else "").strip() for key, column in chosen.items()}
            for key in ("alpha", "beta", "peptide"):
                cleaned[key] = "".join(cleaned[key].split()).upper()
                if set(cleaned[key]) - AA:
                    raise ValueError("Non-standard amino acid in " + key + " at row " + str(number))
            if any(not cleaned[key] for key in required):
                raise ValueError("Missing required value at row " + str(number))
            hla = cleaned["mhc"].upper().replace("HLA-", "").replace(" ", "")
            match = re.fullmatch(r"([ABC])\*?(\d{2}):?(\d{2})", hla)
            if not match:
                raise ValueError("Use a two-field class-I HLA allele, e.g. HLA-A*02:01, at row " + str(number))
            cleaned["mhc"] = "HLA-{}*{}:{}".format(*match.groups())
            rows.append(cleaned)
    if not rows:
        raise ValueError("CSV contains no candidates.")
    return rows


def docker_path():
    executable = shutil.which("docker")
    desktop = Path("/Applications/Docker.app/Contents/Resources/bin/docker")
    if not executable and desktop.is_file():
        executable = str(desktop)
    if not executable:
        raise ValueError("Docker is not installed or is not on PATH.")
    return executable


def mount_arg(source, target, readonly=False):
    output = io.StringIO()
    csv.writer(output, lineterminator="").writerow(
        ["type=bind", "src=" + str(source), "dst=" + target] + (["readonly"] if readonly else [])
    )
    return output.getvalue()


def image_parts():
    """Validate the bundled image before streaming its parts into Docker."""
    manifest = ROOT / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise ValueError("The download is incomplete: SHA256SUMS.txt is missing.")
    records = []
    for line in manifest.read_text().splitlines():
        fields = line.split()
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise ValueError("Invalid image checksum file.")
        if not re.fullmatch(r"images/" + re.escape(ARCHIVE) + r"\.part-\d{2}", fields[1]):
            raise ValueError("Unexpected file in image checksum list.")
        records.append((fields[1], fields[0]))
    records.sort()
    names = [name for name, _ in records]
    expected = ["images/" + ARCHIVE + ".part-{:02d}".format(i) for i in range(len(records))]
    if not records or names != expected:
        raise ValueError("The image parts are missing or out of sequence.")
    parts = []
    print("Checking bundled image...", flush=True)
    for name, expected_hash in records:
        path = ROOT / name
        if not path.is_file():
            raise ValueError("The download is incomplete: " + name + " is missing.")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            first = handle.read(1024 * 1024)
            if first.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Image files are Git LFS pointers, not the image. "
                                 "Use git lfs pull, or download a ZIP with LFS objects included.")
            digest.update(first)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError("Image checksum mismatch: " + name + ". Download the package again.")
        parts.append(path)
    return parts


def load_bundled_image(docker):
    parts = image_parts()
    process = subprocess.Popen([docker, "load"], stdin=subprocess.PIPE)
    try:
        for path in parts:
            with path.open("rb") as handle:
                shutil.copyfileobj(handle, process.stdin, length=1024 * 1024)
        process.stdin.close()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, process.args)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", type=Path, help="Input CSV")
    result.add_argument("--mode", type=mode_name, default="cdr3", help="cdr3/cdr or vregion/v (default: cdr3)")
    result.add_argument("--output", type=Path, help="New or empty results directory")
    result.add_argument("--predictors", nargs="+", default=["all"], help="Predictor names or all; commas also accepted")
    result.add_argument("--threads", type=positive_int, default=2, help="CPU threads per predictor (default: 2)")
    result.add_argument("--keep-logs", action="store_true", help="Keep predictor logs after a successful run")
    result.add_argument("--list-predictors", action="store_true")
    result.add_argument("--load-image", type=Path, help="Load the provided Docker image archive")
    result.add_argument("--image", default=IMAGE, help="Local Docker image tag")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.list_predictors:
            for item in method_config():
                print(item["name"] + "\t" + ", ".join(item["allow_modes"]))
            return 0
        if not args.load_image and (not args.input or not args.output):
            raise ValueError("Provide --input and --output, or use --list-predictors / --load-image.")
        if bool(args.input) != bool(args.output):
            raise ValueError("Provide --input and --output together.")
        methods = select_methods(args.predictors, args.mode)
        if os.environ.get("TCRANK_IN_CONTAINER") == "1":
            if args.load_image or args.image != IMAGE:
                raise ValueError("--load-image and --image are host-only options.")
            sys.path.insert(0, str(ROOT / "core"))
            from cli import main as run_pipeline
            pipeline_args = ["--input", str(args.input), "--output", str(args.output),
                             "--mode", args.mode, "--threads", str(args.threads),
                             "--predictors", *methods]
            if args.keep_logs:
                pipeline_args.append("--keep-logs")
            return run_pipeline(pipeline_args)
        if args.input:
            input_path = args.input.expanduser().resolve(strict=True)
            rows = read_input(input_path, args.mode)
            output = args.output.expanduser().resolve()
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError("Output must be a new or empty directory; existing results are never overwritten.")
        docker = docker_path()
        subprocess.run([docker, "info", "--format", "{{.OSType}}"], check=True, stdout=subprocess.DEVNULL)
        archive = args.load_image
        if archive:
            archive = archive.expanduser().resolve(strict=True)
        elif subprocess.run([docker, "image", "inspect", args.image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            bundled = ROOT / ARCHIVE
            if args.image == IMAGE and bundled.is_file():
                archive = bundled
            elif args.image == IMAGE and (ROOT / "images").is_dir():
                load_bundled_image(docker)
            else:
                raise ValueError("Image is not installed and bundled image files are missing. "
                                 "Download the complete package, or use --load-image with a local archive.")
        if archive:
            subprocess.run([docker, "load", "-i", str(archive)], check=True)
        if not args.input:
            return 0
        subprocess.run([docker, "image", "inspect", args.image], check=True, stdout=subprocess.DEVNULL)
        output.mkdir(parents=True, exist_ok=True)
        name = "tcrank-cli-" + uuid.uuid4().hex[:12]
        command = [docker, "run", "--rm", "--init", "--name", name, "--platform", "linux/amd64",
                   "--network", "none", "--security-opt", "no-new-privileges:true", "--cap-drop", "ALL",
                   "--mount", mount_arg(input_path, "/input/candidates.csv", True),
                   "--mount", mount_arg(output, "/output")]
        if hasattr(os, "getuid"):
            command += ["--user", str(os.getuid()) + ":" + str(os.getgid())]
        command += [args.image, "--input", "/input/candidates.csv", "--output", "/output",
                    "--mode", args.mode, "--threads", str(args.threads), "--predictors", *methods]
        if args.keep_logs:
            command.append("--keep-logs")
        print("TCRank: {} candidates; {} mode; {} predictors".format(len(rows), args.mode, len(methods)), flush=True)
        try:
            completed = subprocess.run(command)
        except KeyboardInterrupt:
            subprocess.run([docker, "stop", "--time", "5", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Cancelled. No background prediction job remains.", file=sys.stderr)
            return 130
        print("Results: " + str(output), flush=True)
        return completed.returncode
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
