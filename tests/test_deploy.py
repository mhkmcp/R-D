"""fidnn.deploy: the exportable twin matches the fidnn pipeline, in torch and through ONNX."""

import numpy as np
import pytest
import torch
from sklearn.svm import OneClassSVM

from fidnn.deploy.graph import MonitoredClassifier, tap_features
from fidnn.detect.features import CleanFeatures, add_block_e
from fidnn.detect.normalise import RobustNormaliser
from fidnn.models.registry import build
from fidnn.taps.features import FEATURE_NAMES
from fidnn.taps.features import tap_features as reference
from fidnn.taps.hooks import TapMonitor
from fidnn.taps.registry import taps


@pytest.mark.parametrize("shape", [(4, 16, 8, 8), (4, 64), (4, 10)])
def test_twin_descriptor_matches_reference(shape):
    a = torch.randn(*shape)
    if len(shape) == 4:
        a = a.relu()
        a[:, 3] = 0  # a dead channel: the entropy term must stay finite
    torch.testing.assert_close(tap_features(a, 0.5), reference(a, 0.5), rtol=1e-4, atol=1e-5)


@pytest.fixture(scope="module")
def graph():
    torch.manual_seed(0)
    model = build("m1").eval()
    tap_list = taps("m1")
    x = torch.randint(0, 256, (64, 32, 32, 3), dtype=torch.uint8)
    xn = x.permute(0, 3, 1, 2).float() / 255
    with torch.no_grad(), TapMonitor(model, tap_list) as mon:
        model(xn)
        v = torch.stack([mon.outputs[t.tap_id] for t in tap_list], 1).numpy()
    clean = CleanFeatures(add_block_e(v, FEATURE_NAMES), [t.tap_id for t in tap_list],
                          FEATURE_NAMES + ["e1", "e2", "e3"])
    norm = RobustNormaliser().fit(clean)
    z = norm.transform(clean).reshape(len(clean), -1)
    svm = OneClassSVM(kernel="rbf", nu=0.1, gamma=0.01).fit(z)
    g = MonitoredClassifier(build("m1").eval(), [t.path for t in tap_list],
                            [float("inf")] * len(tap_list), [0, 0, 0], [1, 1, 1], norm, svm,
                            [0.0])
    g.model.load_state_dict(model.state_dict())
    return g, x, -svm.decision_function(z)


def test_graph_score_is_svdd_score(graph):
    g, x, expected = graph
    with torch.no_grad():
        score = g(x)[3].numpy()
    np.testing.assert_allclose(score, expected, rtol=1e-4, atol=1e-4)


def test_onnx_round_trip(graph, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    g, x, expected = graph
    path = tmp_path / "m.onnx"
    torch.onnx.export(g, (x[:4],), path, input_names=["images"],
                      dynamic_shapes={"images": {0: torch.export.Dim("batch")}},
                      dynamo=True, external_data=False, verbose=False)
    out = ort.InferenceSession(str(path)).run(None, {"images": x.numpy()})
    np.testing.assert_allclose(out[3], expected, rtol=1e-3, atol=1e-3)
    assert out[4].shape == (len(x), 1)


def test_classifier_layouts_agree():
    from fidnn.deploy.formats import Classifier

    torch.manual_seed(0)
    model = build("m1").eval()
    x = torch.randint(0, 256, (4, 32, 32, 3), dtype=torch.uint8)
    mean, std = [0.5, 0.4, 0.3], [0.2, 0.25, 0.3]
    nhwc = Classifier(model, mean, std)(x)
    nchw = Classifier(model, mean, std, nhwc_uint8=False)(x.permute(0, 3, 1, 2).float())
    xn = (x.permute(0, 3, 1, 2).float() / 255 - torch.tensor(mean).view(1, 3, 1, 1)) \
        / torch.tensor(std).view(1, 3, 1, 1)
    torch.testing.assert_close(nhwc, model(xn), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(nchw, nhwc, rtol=1e-4, atol=1e-4)
