"""SPEC §0 C1, §11.4: no Mahalanobis, covariance estimators, ZCA or whitening in src/ or configs/."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED = re.compile(
    r"mahalanobis|EllipticEnvelope|MinCovDet|EmpiricalCovariance|LedoitWolf|ShrunkCovariance"
    r"|\bOAS\b|GraphicalLasso|zca|whiten\s*=\s*True|(inv|pinv)\s*\(\s*(np|numpy|torch)\.cov",
    re.IGNORECASE,
)


def _files():
    for sub in ("src", "configs"):
        for p in (ROOT / sub).rglob("*"):
            if p.suffix in {".py", ".yaml", ".yml", ".toml", ".json"}:
                yield p


def test_no_banned_constructs():
    hits = [
        f"{p.relative_to(ROOT)}:{i}: {line.strip()}"
        for p in _files()
        for i, line in enumerate(p.read_text().splitlines(), start=1)
        if BANNED.search(line)
    ]
    assert not hits, "C1 violations:\n" + "\n".join(hits)


def test_guard_catches_known_violations():
    for bad in ("from sklearn.covariance import MinCovDet", "PCA(whiten=True)",
                "np.linalg.inv(np.cov(x))", "zca_whiten(x)", "mahalanobis(u, v, vi)"):
        assert BANNED.search(bad), bad
