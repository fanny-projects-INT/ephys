# Ephys Pipeline

Pipeline to process Neuropixels recordings using SpikeInterface and Kilosort4.

## Overview

This pipeline runs the following steps for each session:

1. Compress raw `.bin` files → `.cbin`
2. Load recordings
3. Preprocess signals
4. Run spike sorting (Kilosort4)
5. Compute sorting analyzers
6. Export results to ALF format (IBL compatible)

---

## Installation

Create a Python environment :

```bash
python -m venv ephys_env
source ephys_env/bin/activate  # Linux / Mac
ephys_env\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data structure

Expected structure:

```
DATA_ROOT/
└── Mouse_ID/
    └── YYYY_MM_DD/
        ├── Rec/
        │   └── probe00/
        │       ├── *.ap.bin / *.ap.cbin
        │       ├── *.lf.bin / *.lf.cbin
        │
        ├── KS/
        ├── alf/
        └── sorting_analyzer/
```

- Rec: raw SpikeGLX data  
- KS: Kilosort outputs  
- alf: exported ALF files  
- sorting_analyzer: SpikeInterface analyzer outputs  

---

## Usage

Edit `main.py`:

```python
DATA_ROOT = Path("F:/Data_Mice_IBL")

SESSION_LIST = [
    "VF074_2026_03_24",
]
```

Run:

```bash
python main.py
```

---

## Pipeline steps

- compress_recordings  
  Compress .bin files to .cbin

- load_recordings  
  Load AP data

- preprocess_recordings  
  Phase shift, bandpass (300–6000 Hz), bad channel detection, interpolation, referencing

- run_kilosort4  
  Run spike sorting

- build_sorting_analyzers  
  Compute waveforms, templates, amplitudes, metrics

- export_alf  
  Export to ALF (IBL format)

---

## Notes

- Supports probe00 and probe01
- Outputs can be overwritten depending on parameters
- Errors are handled per probe