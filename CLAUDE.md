# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A handwritten digit (MNIST) recognizer, split into two independent versions that share one trained model:

- [desktop_version/](desktop_version/) — the original PyTorch project: training (`train.py`), image-file CLI (`predict.py`), Tkinter drawing GUI (`app.py`), and `export_weights.py`, which converts the trained `mnist_cnn.pt` for the web version. See [desktop_version/CLAUDE.md](desktop_version/CLAUDE.md).
- [web_version/](web_version/) — a static page that runs the same CNN in pure JavaScript (no libraries, no build step), deployed to GitHub Pages. See [web_version/CLAUDE.md](web_version/CLAUDE.md).

All code, comments, and UI text are written in Korean.

## How the two versions connect

The web version never runs Python. Its weights (`web_version/model/`) and test fixtures (`web_version/tests/fixtures/`) are generated files committed to git:

```
desktop_version/train.py          -> desktop_version/mnist_cnn.pt
desktop_version/export_weights.py -> web_version/model/weights.{bin,json}
                                   + web_version/tests/fixtures/{reference.json, sample_*.png}
```

So a retrain must be followed by `export_weights.py` and a run of the web tests. The model architecture (`desktop_version/model.py` ↔ `web_version/js/model.js`) and the preprocessing (`train.py`/`predict.py`/`app.py` ↔ `web_version/js/preprocess.js`) are duplicated across languages and must be kept in sync by hand.

## Environment

- No Python interpreter is on `PATH`; use `C:\Users\<username>\AppData\Local\Programs\Python\Python312\python.exe`. Dependencies (no `requirements.txt`): `torch`, `torchvision` (CPU build), `pillow`, `numpy`.
- No Node.js; web tests run in a browser (`web_version/tests/test.html`).

## Deployment

`.github/workflows/pages.yml` deploys the `web_version/` folder to GitHub Pages on pushes to `master` that touch it (or via manual dispatch). The repository's Pages source must be set to "GitHub Actions".
