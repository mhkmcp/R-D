"""Baselines H1 is tested against (SPEC §6.4). C1: no covariance-based score anywhere here."""

from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_softmax, logsumexp
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity, LocalOutlierFactor

from fidnn.detect.features import CleanFeatures, Features
from fidnn.detect.normalise import RobustNormaliser

OUTPUT_ONLY = ("msp", "entropy", "margin", "energy")
FEATURE_SPACE = ("isolation_forest", "lof", "kde", "pca_recon")


def output_only_scores(logits: np.ndarray) -> dict[str, np.ndarray]:
    """Higher means more anomalous, so the SVDD convention holds for every baseline."""
    logp = log_softmax(logits, axis=1)
    p = np.exp(logp)
    top2 = np.sort(logits, axis=1)[:, -2:]
    return {
        "msp": -p.max(axis=1),
        "entropy": -(p * logp).sum(axis=1),
        "margin": -(top2[:, 1] - top2[:, 0]),
        "energy": -logsumexp(logits, axis=1),
    }


@dataclass
class FeatureSpaceBaselines:
    """Same normalised D2 features, so the comparison isolates the detector (§6.4)."""

    seed: int = 0
    pca_components: int = 16
    normaliser: RobustNormaliser = field(default_factory=RobustNormaliser)
    models: dict = field(default_factory=dict, repr=False)

    def fit(self, clean_fit: CleanFeatures) -> "FeatureSpaceBaselines":
        if not isinstance(clean_fit, CleanFeatures):
            raise TypeError("baselines fit on clean_fit only (C2)")
        self.normaliser.fit(clean_fit)
        z = self.normaliser.transform(clean_fit).reshape(len(clean_fit), -1)
        self.models["isolation_forest"] = IsolationForest(random_state=self.seed).fit(z)
        self.models["lof"] = LocalOutlierFactor(novelty=True).fit(z)
        self.models["kde"] = KernelDensity().fit(z)
        # reconstruction error only; the scaling flag stays off (C1 — see detect/README.md)
        self.models["pca_recon"] = PCA(n_components=min(self.pca_components, z.shape[1]),
                                       whiten=False, random_state=self.seed).fit(z)
        return self

    def scores(self, features: Features) -> dict[str, np.ndarray]:
        z = self.normaliser.transform(features).reshape(len(features), -1)
        pca = self.models["pca_recon"]
        recon = pca.inverse_transform(pca.transform(z))
        return {
            "isolation_forest": -self.models["isolation_forest"].decision_function(z),
            "lof": -self.models["lof"].decision_function(z),
            "kde": -self.models["kde"].score_samples(z),
            "pca_recon": ((z - recon) ** 2).sum(axis=1),
        }
