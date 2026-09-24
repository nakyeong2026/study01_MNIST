# CLAUDE.md — web_version

Guidance for Claude Code when working in `web_version/`. See the root [CLAUDE.md](../CLAUDE.md) for how this relates to `desktop_version/`.

## What this is

A static web page that recognizes handwritten digits entirely in the browser: the user draws on a canvas (mouse or touch) or uploads an image, and a hand-written pure-JavaScript implementation of the same CNN as `../desktop_version/model.py` runs inference. No libraries, CDNs, npm, bundler, or build step — every file here is served as-is. Deployed to GitHub Pages by `../.github/workflows/pages.yml` (repo Settings → Pages → Source must be "GitHub Actions"). All code comments and UI text are in Korean.

## Commands

The page uses ES modules and `fetch`, so it does not work from `file://`. Serve it over HTTP from the repo root (Python is not on `PATH`; see `../desktop_version/CLAUDE.md` for the interpreter path):

```
python -m http.server 8000 --directory web_version
```

- App: `http://localhost:8000/`
- Tests: `http://localhost:8000/tests/test.html` — the page title becomes `PASS n/n` or `FAIL k/n`. Hard-reload (Ctrl+Shift+R) if results look stale.

## Architecture

- `js/ops.js` — pure tensor ops on row-major `Float32Array`s: `conv2d3x3` (stride 1, padding 1), `relu`, `maxPool2x2`, `linear`, `softmax`, `argmax`.
- `js/model.js` — `parseWeights`/`loadWeights` read `model/weights.json` + `model/weights.bin`; `forward` hard-codes the `MnistCNN` layer sequence (dropout omitted, it is a no-op at inference); `predict` returns `{digit, confidence, probabilities}`. If `../desktop_version/model.py` changes, this file must change with it.
- `js/preprocess.js` — mirrors the desktop pipelines: `preprocessDrawing` (280×280 canvas → area-average to 28×28, approximating `app.py`'s LANCZOS resize) and `preprocessPhoto` (PIL-`convert("L")` grayscale → invert if mean > 127 → aspect-preserving pad to 28×28, approximating `predict.py`'s `ImageOps.pad` BICUBIC resize), both then normalized with mean 0.1307 / std 0.3081.
- `js/app.js` — DOM wiring: Pointer Events drawing (pen width 18, recognizes on stroke end), file upload, probability bars, 28×28 input preview.
- `model/` — **generated, do not edit by hand.** Regenerate with `python ../desktop_version/export_weights.py` after every retrain; it also rewrites `tests/fixtures/`.
- `tests/` — tiny in-house harness (`harness.js`); `run-tests.js` imports every `*.test.js`, so add an import there for a new test file. `parity.test.js` checks JS logits against PyTorch's (`fixtures/reference.json`) and `preprocess.test.js` checks decoded sample PNGs against PyTorch's preprocessed inputs — these are the guards that the JS port matches the Python model.
