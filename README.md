# Ephys Pipeline

Pipeline to process Neuropixels recordings using SpikeInterface, Kilosort4, Bombcell, and ALF export.

## Overview

This pipeline runs the following steps for each session:

1. Compress raw .bin files to .cbin
2. Load AP recordings
3. Preprocess signals
4. Run spike sorting with Kilosort4
5. Compute sorting analyzers
6. Run Bombcell quality control
7. Export results to ALF format (IBL compatible)
8. Compute and save behavior / ephys alignment

---

## Installation

Create a Python environment:

python -m venv ephys_env
source ephys_env/bin/activate   (Linux / Mac)
ephys_env\Scripts\activate      (Windows)

Install dependencies:

pip install -r requirements.txt

---

## Configuration

Create a config.py file from the template:

cp config_template.py config.py

Then edit it:

from pathlib import Path
DATA_ROOT = Path(r"F:\Data_Mice_IBL")
DB_PATH = Path(r"F:\Data_Mice_IBL\full_db_all_rigs.feather")

Optional parameters:

KS_PARAMS = {...}
COMPRESS_KEEP_ORIGINAL = True
KS_REMOVE_EXISTING_FOLDER = True

ANALYZER_N_JOBS = 8
ANALYZER_CHUNK_DURATION = "1s"

EXPORT_N_JOBS = 8
EXPORT_CHUNK_DURATION = "1s"

---

## Data structure

Only the Rec/ folder is required at the beginning. All other folders are created automatically.

DATA_ROOT/
└── Mouse_ID/
    └── YYYY_MM_DD/
        ├── Rec/
        │   └── probe00/
        │       ├── *.ap.bin / *.ap.cbin
        │       ├── *.lf.bin / *.lf.cbin
        │       ├── *.ap.meta
        │       └── *.lf.meta
        │
        ├── KS/
        │   └── probe00/
        ├── sorting_analyzer/
        │   └── probe00/
        ├── bombcell/
        │   └── probe00/
        ├── alf/
        │   └── probe00/
        ├── shift.txt
        └── alignment_affine.json

---

## Folder description

Rec: raw or compressed SpikeGLX recordings  
KS: Kilosort4 outputs  
sorting_analyzer: SpikeInterface analyzer outputs  
bombcell: Bombcell QC outputs  
alf: exported ALF files for IBL GUI  
shift.txt: alignment offset (b)  
alignment_affine.json: full affine transform (a, b)

---

## Session naming

Sessions must follow:

Mouse_ID_YYYY_MM_DD

Example:
VF074_2026_03_24

The pipeline automatically extracts:
- mouse ID
- date

Behavior and ephys must use the same Mouse_ID.

---

## Usage

Edit main.py:

SESSION_LIST = [
    "VF074_2026_03_24",
]

Run:

python main.py

---

## Pipeline steps

compress_recordings  
→ Compress .bin files to .cbin  

load_recordings  
→ Load AP recordings  

preprocess_recordings  
→ Phase shift, bandpass (300–6000 Hz), bad channel detection, interpolation, common referencing  

run_kilosort4  
→ Run spike sorting  

build_sorting_analyzers  
→ Compute waveforms, templates, amplitudes, metrics  

run_bombcell  
→ Classify units (good / mua / noise)  

export_alf  
→ Export to ALF format (IBL compatible)  

compute_and_save_alignment  
→ Compute alignment between behavior and ephys using LF sync channel  
→ Saves shift.txt and alignment_affine.json  

---

## Notes

- Supports probe00 and probe01  
- Works with compressed (.cbin) files  
- Alignment uses LF sync channel  
- Alignment assumes same mouse name in behavior DB and ephys folder  
- Outputs can be overwritten depending on parameters  
- Errors are handled per probe  
