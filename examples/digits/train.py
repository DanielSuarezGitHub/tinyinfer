"""Train a small digit classifier. Requires NumPy."""
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import numpy as np

folder = Path(__file__).parent
folder.joinpath("data").mkdir(exist_ok=True)
archive = folder / "data/digits.zip"
if not archive.exists():
    urlretrieve(
        "https://archive.ics.uci.edu/static/public/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits.zip",
        archive,
    )
with zipfile.ZipFile(archive) as files:
    train = np.loadtxt(files.open("optdigits.tra"), delimiter=",")
    test_bytes = files.read("optdigits.tes")

x = (train[:, :64] / 16).astype(np.float32)
y = train[:, 64].astype(int)
rng = np.random.default_rng(42)
w1 = (rng.standard_normal((64, 64)) * np.sqrt(2 / 64)).astype(np.float32)
b1 = np.zeros(64, dtype=np.float32)
w2 = (rng.standard_normal((64, 10)) * np.sqrt(2 / 64)).astype(np.float32)
b2 = np.zeros(10, dtype=np.float32)

for epoch in range(200):
    order = rng.permutation(len(x))
    for start in range(0, len(x), 128):
        batch = order[start:start + 128]
        inputs, labels = x[batch], y[batch]
        hidden = np.maximum(inputs @ w1 + b1, 0)
        scores = hidden @ w2 + b2
        probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        # Gradient of average softmax cross-entropy.
        probabilities[np.arange(len(batch)), labels] -= 1
        probabilities /= len(batch)
        hidden_gradient = (probabilities @ w2.T) * (hidden > 0)
        dw1 = inputs.T @ hidden_gradient
        db1 = hidden_gradient.sum(axis=0)
        dw2 = hidden.T @ probabilities
        db2 = probabilities.sum(axis=0)

        # Plain stochastic gradient descent.
        w1 -= 0.1 * dw1
        b1 -= 0.1 * db1
        w2 -= 0.1 * dw2
        b2 -= 0.1 * db2

with (folder / "model.txt").open("w") as file:
    file.write("64 64 10\n")
    for weights in [w1, b1, w2, b2]:
        np.savetxt(file, weights.reshape(1, -1), fmt="%.9g")
(folder / "test.csv").write_bytes(test_bytes)
test = np.loadtxt(folder / "test.csv", delimiter=",")
hidden = np.maximum((test[:, :64] / 16).astype(np.float32) @ w1 + b1, 0)
predictions = (hidden @ w2 + b2).argmax(axis=1)
print(f"Test accuracy: {(predictions == test[:, 64]).mean():.2%}")
