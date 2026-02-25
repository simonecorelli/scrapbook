# CLAUDE.md

This file provides guidance to AI assistants working with this repository.

## Repository Overview

**scrapbook** is an educational research repository by Paolo Prandoni (LCAV, EPFL), mirrored from [github.com/prandoni/scrapbook](https://github.com/prandoni/scrapbook). It contains self-contained Jupyter notebooks exploring signal processing and machine learning concepts.

## Repository Structure

```
scrapbook/
├── PeakRMS.ipynb          # Audio RMS/peak analysis and filtering
├── pinknoise.ipynb        # Pink noise spectral analysis and octave band modeling
├── ergodic.ipynb          # Stochastic processes, ergodicity, and linear filtering
├── faceAligment/          # Face alignment project (note: directory name is misspelled)
│   ├── faceAlignment.ipynb  # Cascade regression tree face alignment implementation
│   ├── faces.jpg
│   ├── landmarks_68.png
│   ├── face_shape_model.gif
│   └── ...                # Reference images and diagrams
├── requirements.txt       # pip dependencies
└── environment.yml        # Conda environment specification
```

## Technology Stack

- **Language:** Python 2 (notebooks use Python 2.7.11 kernel)
- **Environment:** Jupyter notebooks (`.ipynb`)
- **Core libraries:** NumPy, SciPy, Matplotlib, IPython
- **Presentation:** RISE (Reveal.js slideshow extension for Jupyter)

## Environment Setup

Two equivalent options for setting up the environment:

**Option A — Conda (recommended, includes RISE for slideshows):**
```bash
conda env create -f environment.yml
conda activate rise-environment
jupyter notebook
```

**Option B — pip:**
```bash
pip install -r requirements.txt
jupyter notebook
```

## Notebooks Summary

| Notebook | Topic | Key Libraries |
|---|---|---|
| `PeakRMS.ipynb` | Audio analysis: RMS extraction, peak detection, dB normalization | `scipy.signal`, `scipy.io.wavfile` |
| `pinknoise.ipynb` | Spectral power density, octave band filter banks (1/1 and 1/3 octave) | `numpy`, `matplotlib` |
| `ergodic.ipynb` | Ergodic vs. non-ergodic processes, IID random processes, linear filtering | `numpy`, `scipy.signal` |
| `faceAligment/faceAlignment.ipynb` | Cascade regression trees for 68-point face landmark detection | `numpy`, `matplotlib` |

## Code Conventions

### Naming
- **Classes:** PascalCase — `RegTree`, `RegForest`, `RCascade`, `DataObject`, `TrainingSample`
- **Functions/methods:** snake_case — `train`, `regress`, `estimate`, `test_split`
- **Constants:** UPPER_SNAKE_CASE — `WIN_LEN_MS`, `TRIALS_PER_NODE`, `POINTS`
- **Math variables:** single-letter names following domain conventions (`x`, `y`, `f`, `t`, `p`)

### Python 2 compatibility
The notebooks use Python 2 idioms:
- `xrange()` instead of `range()`
- Python 2 `print` statements (not functions in some cells)

### Notebook Structure
Each notebook follows a consistent pattern:
1. Markdown cells explaining theory with LaTeX equations
2. Code cells implementing algorithms from scratch
3. Visualization cells validating results through plots
4. Reference images (PNG/GIF/JPG) embedded for context

### Algorithm Style
- Custom implementations over library abstractions (prioritizes pedagogy)
- Explicit algorithm steps (not hidden behind wrappers)
- Synthetic data generation for reproducibility
- NumPy vectorization where appropriate

## No Testing Infrastructure

There are no automated tests, CI/CD pipelines, or linters. The notebooks themselves serve as the primary validation mechanism through visualization and inline output. When modifying notebooks, ensure all cells execute cleanly from top to bottom (Kernel → Restart & Run All).

## Key Algorithms (faceAlignment.ipynb)

The face alignment notebook implements the cascade regression approach from:
> "One Millisecond Face Alignment with an Ensemble of Regression Trees" — Kazemi & Sullivan, KTH (2014)

Main classes:
- `RegTree` — Single regression tree with greedy split optimization
- `RegForest` — Boosted ensemble of shallow `RegTree` instances
- `RCascade` — Multi-stage cascade of `RegForest` regressors
- `DataObject` / `TrainingSample` — Data containers

Pipeline: single tree → random forest → cascade with shape-invariant pixel-difference features and shape normalization (warping) between stages.

## Git Workflow

- **Main branch:** `master`
- **Working directory:** should be clean before starting; commit messages have been brief/informal historically (e.g., "improved", "Add files via upload")
- No branch protection rules or PR requirements have been observed

## Notes for AI Assistants

- This is a **read-and-learn** repository; changes should preserve the educational narrative flow of each notebook.
- Notebook cells are ordered intentionally — do not reorder them without understanding the pedagogical progression.
- The `faceAligment/` directory name is intentionally left as-is (misspelled) to match the existing structure; do not rename it.
- External data files referenced in notebooks (e.g., `beethoven.wav`) are not included in the repository.
- When adding or editing notebooks, maintain the existing LaTeX/markdown documentation style.
