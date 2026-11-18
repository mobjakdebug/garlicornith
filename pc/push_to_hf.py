"""Put the model on Hugging Face: the weights, the two indexes, the card and a screenshot.

    HF_TOKEN=hf_… python pc/push_to_hf.py              # everything (about 4.8 GiB the first time)
    HF_TOKEN=hf_… python pc/push_to_hf.py --card-only  # just README.md, .gitattributes and the images

The repository is `officialmobai/station-9b` unless HF_REPO says otherwise; it is created if it is not there. The
weights are what `pc/ornith_convert.py` writes into models/, so this has to run after the conversion (in CI it runs in
the package job, after both packages are built).

What it does not do: it never writes the token anywhere, and it refuses to push the weights when they are not there,
rather than creating an empty repository that a console would then fail to download from.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = os.environ.get("HF_REPO") or "officialmobai/station-9b"

# (what is here, what it is called in the repository)
CARD = [
    ("hf/README.md", "README.md"),
    ("hf/.gitattributes", ".gitattributes"),
    ("docs/ui/chat.png", "screenshots/chat.png"),
    ("docs/ui/mark.png", "logo.png"),
]
MODEL = [
    ("chat/assets_9b/model.idx", "model.idx"),
    ("chat/assets_9b/mtp.idx", "mtp.idx"),
    ("models/ornith-1.5-9b-q4s.bin", "ornith-1.5-9b-q4s.bin"),
    ("models/ornith-1.5-9b-mtp.bin", "ornith-1.5-9b-mtp.bin"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card-only", action="store_true", help="the card and the images, not the weights")
    ap.add_argument("--repo", default=REPO, help=f"default: {REPO}")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("HF_TOKEN is not set: nothing to push with")
        return 0                                    # not an error: the workflow runs without the secret too

    wanted = CARD + ([] if args.card_only else MODEL)
    files = [(ROOT / src, dst) for src, dst in wanted]
    missing = [str(p) for p, _ in files if not p.exists()]
    if missing:
        print("not pushing: these are not here:", *missing, sep="\n  ")
        return 1
    for p, dst in files:
        print(f"  {dst:28s} {p.stat().st_size:>13,d} B  <- {p.relative_to(ROOT)}")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed: pip install huggingface_hub")
        return 1

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo, repo_type="model", exist_ok=True, private=False)
    for p, dst in files:
        print(f"uploading {dst} ({p.stat().st_size:,} B)…", flush=True)
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dst, repo_id=args.repo, repo_type="model",
                        commit_message=f"Station 9B: {dst}")
    print(f"https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
