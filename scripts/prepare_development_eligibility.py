"""Write an explicit amended source certificate; never modify the original audit."""

import argparse
from pathlib import Path

from curiosity_study.artifacts import digest, write_once
from source_admission import build_certificate


def prepare(source_collection, source_audit, amendment, out):
    certificate = build_certificate(source_collection, source_audit, amendment)
    write_once(out, certificate)
    print(dict(path=str(out.resolve()), sha256=digest(certificate)))
    return certificate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-collection", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_collection, args.source_audit, args.amendment, args.out)
