from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from html import unescape
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen


USER_AGENT = "Mozilla/5.0"


def download_url(url: str) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        content = response.read()
        content_type = response.headers.get_content_type()
    return content, content_type


def resolve_onedrive_download(url: str) -> str:
    content, content_type = download_url(url)
    if content_type != "text/html":
        return url

    html = content.decode("utf-8", errors="ignore")
    patterns = [
        r'"downloadUrl":"([^"]+)"',
        r'"@content\\.downloadUrl":"([^"]+)"',
        r'"url":"([^"]+download[^"]+)"',
        r'href="([^"]+download=1[^"]*)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return unescape(match.group(1)).replace("\\u0026", "&").replace("\\/", "/")

    if "download=1" not in url:
        return url + ("&" if "?" in url else "?") + "download=1"
    return url


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_url = resolve_onedrive_download(url) if "1drv.ms" in url else url
    content, _ = download_url(resolved_url)
    if content.lstrip().startswith(b"<"):
        raise RuntimeError(f"download returned HTML instead of weights: {resolved_url}")
    destination.write_bytes(content)


def ensure_spacy_model(model_name: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", model_name],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare non-Git assets for the Linux server workflow.")
    parser.add_argument(
        "--manifest",
        default="tools/assets_manifest.json",
        help="Path to the asset manifest JSON.",
    )
    parser.add_argument(
        "--weights-dir",
        default="data/weights",
        help="Directory where the backbone weights should be stored.",
    )
    parser.add_argument(
        "--skip-spacy",
        action="store_true",
        help="Skip the spaCy model download step.",
    )
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    weights_info = manifest["backbone_weights"]
    weights_path = Path(args.weights_dir) / weights_info["filename"]
    if weights_path.exists():
        print(f"weights already present: {weights_path}")
    else:
        print(f"downloading backbone weights to {weights_path}")
        download_file(weights_info["url"], weights_path)

    if not args.skip_spacy:
        ensure_spacy_model(manifest["spacy_model"]["name"])


if __name__ == "__main__":
    main()
