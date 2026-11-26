import numpy as np
from src.bow import BagOfWords

def test_bow_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    desc = [rng.normal(size=(30, 8)).astype("float32") for _ in range(6)]
    bow = BagOfWords(n_clusters=4)
    bow.fit_vocabulary(desc)
    hist = np.vstack([bow.histogram(x) for x in desc])
    assert hist.shape == (6, 4)
    assert np.allclose(np.linalg.norm(hist, axis=1), 1.0)
    bow.fit_classifier(hist, ["a", "a", "a", "b", "b", "b"])
    path = tmp_path / "model.joblib"
    bow.save_model(path, {"name": "sift"})
    loaded = BagOfWords.load_model(path)
    assert loaded.predict(hist[0]) in {"a", "b"}
