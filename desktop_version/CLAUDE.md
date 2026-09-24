# CLAUDE.md — desktop_version

Guidance for Claude Code when working in `desktop_version/`. See the root [CLAUDE.md](../CLAUDE.md) for how this relates to `web_version/`.

## What this is

The original PyTorch project: trains a CNN on MNIST and lets a user try it two ways — by pointing at an image file (`predict.py`) or by drawing with the mouse in a Tkinter GUI (`app.py`). It is also the source of the weights the web version uses (`export_weights.py`). All code and comments are written in Korean.

## Environment

- No Python interpreter is on `PATH`. Use `C:\Users\<username>\AppData\Local\Programs\Python\Python312\python.exe` (installed via `winget install -e --id Python.Python.3.12 --scope user`).
- Dependencies (pip, no `requirements.txt`): `torch`, `torchvision` (CPU build, from `https://download.pytorch.org/whl/cpu`), `pillow`, `numpy`.

## Commands

All scripts resolve `mnist_cnn.pt` and `data/` relative to their own file, so they can be run from any working directory.

```
# Train and (re)write mnist_cnn.pt — downloads MNIST into desktop_version/data on first run
python train.py

# Recognize a digit from an image file
python predict.py <이미지_경로>

# Launch the mouse-drawing GUI (blocks in a Tk mainloop)
python app.py

# Export mnist_cnn.pt for the web version (weights + test fixtures) — rerun after every retrain
python export_weights.py
```

`app.bat` double-click-launches `app.py` via `pythonw.exe` with no console window. It resolves the interpreter via `%USERNAME%` and hardcodes the `Python312` install path.

There is no lint config or test suite; verify changes by running `predict.py` on the `sample_*_label_N.png` files (the filename holds the correct digit).

## Architecture

- `model.py` — `MnistCNN`, the only model definition. `mnist_cnn.pt` is a raw `state_dict` with no architecture metadata, so changing layer shapes invalidates it and requires retraining. The web version reimplements this architecture by hand in `../web_version/js/model.js` — keep both in sync.
- `train.py` — downloads MNIST into `data/`, trains 5 epochs, overwrites `mnist_cnn.pt`.
- `predict.py` — CLI. Preprocessing auto-inverts images whose mean pixel is bright (>127), since MNIST is white-on-black; then pads/resizes to 28×28 and normalizes (mean 0.1307, std 0.3081).
- `app.py` — Tkinter GUI. Draws to the visible `Canvas` and, in parallel, to an in-memory `PIL.Image` at every drag event, so the 28×28 input is built without screen capture. Exits with a message if `mnist_cnn.pt` doesn't exist.
- `export_weights.py` — writes `../web_version/model/weights.bin` (all tensors as little-endian float32, concatenated in `state_dict` order) and `weights.json` (name, shape, offset, length in floats), plus `../web_version/tests/fixtures/` (sample PNGs and `reference.json` with PyTorch inputs/logits the web tests compare against).
- `train.py`, `predict.py`, `app.py` each build the same `Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])`, and `../web_version/js/preprocess.js` mirrors it — keep all of them in sync if preprocessing changes.
