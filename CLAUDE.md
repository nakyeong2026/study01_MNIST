# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small PyTorch project that trains a CNN to recognize handwritten digits (MNIST) and lets a user try it two ways: by pointing at an image file, or by drawing with the mouse in a desktop GUI. All code and comments are written in Korean.

## Environment

- No Python interpreter is on `PATH` in this environment. The interpreter used to set this project up lives at:
  `C:\Users\<username>\AppData\Local\Programs\Python\Python312\python.exe` (installed via `winget install -e --id Python.Python.3.12 --scope user`)
- Dependencies (installed via pip, no `requirements.txt` in the repo): `torch`, `torchvision` (CPU build, from `https://download.pytorch.org/whl/cpu`), `pillow`, `numpy`.
- This directory is not a git repository.

## Commands

Run these with the full path to `python.exe` above (or `pythonw.exe` for the GUI, to avoid a console window), since there's no `python` on `PATH`.

```
# Train the model and (re)write mnist_cnn.pt — downloads MNIST into ./data on first run
python train.py

# Recognize a digit from an image file
python predict.py <이미지_경로>

# Launch the mouse-drawing GUI (blocks in a Tk mainloop)
python app.py
```

`app.bat` double-click-launches `app.py` via `pythonw.exe` with no console window. It resolves the interpreter path via `%USERNAME%`, and hardcodes the `Python312` install path — update it if the interpreter is reinstalled elsewhere.

There is no lint config or test suite in this repo.

## Architecture

- [model.py](model.py) — `MnistCNN`, the only model definition. Both `predict.py` and `app.py` import it and must stay in sync with `train.py`'s architecture, since `mnist_cnn.pt` is a raw `state_dict` (no architecture metadata saved with it) — changing layer shapes in `model.py` invalidates the existing `mnist_cnn.pt` and requires retraining.
- [train.py](train.py) — downloads MNIST via `torchvision.datasets.MNIST` into `./data`, trains for 5 epochs, and overwrites `mnist_cnn.pt` with `model.state_dict()`.
- [predict.py](predict.py) — CLI entry point (`sys.argv[1]` = image path). Preprocessing auto-inverts images whose mean pixel value is bright (>127), since MNIST is white-digit-on-black and photos of pen-on-paper are the opposite; then pads/resizes to 28×28 and applies MNIST's standard normalization (mean 0.1307, std 0.3081) before inference.
- [app.py](app.py) — Tkinter GUI. Draws to the visible `Canvas` and, in parallel, to an in-memory `PIL.Image` via `ImageDraw` at every mouse-drag event, so the 28×28 model input is built without any screen capture. Exits with a message if `mnist_cnn.pt` doesn't exist yet (i.e. `train.py` hasn't been run).
- Both `predict.py` and `app.py` independently reconstruct the same `transforms.Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])` pipeline used in `train.py` — keep all three in sync if preprocessing changes.
