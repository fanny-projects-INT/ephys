from pathlib import Path
from config import DATA_ROOT, DB_PATH
from spikeinterface.sorters import get_default_sorter_params
from functions.paths import build_paths
from functions.compress import compress_recordings
from functions.load import load_recordings
from functions.preprocess import preprocess_recordings
from functions.sort import run_kilosort4
from functions.analyzer import build_sorting_analyzers
from functions.alf import export_alf


# =========================================================
# CONFIG
# =========================================================

SESSION_LIST = [
    "VF074test_2026_03_24",
]

KS_PARAMS = get_default_sorter_params("kilosort4")

COMPRESS_KEEP_ORIGINAL = True
KS_REMOVE_EXISTING_FOLDER = True
ANALYZER_N_JOBS = 1
ANALYZER_CHUNK_DURATION = "1s"
EXPORT_N_JOBS = 1
EXPORT_CHUNK_DURATION = "1s"


# =========================================================
# MAIN
# =========================================================
def main():
    sessions = [
        build_paths(session_name, data_root=DATA_ROOT, db_path=DB_PATH)
        for session_name in SESSION_LIST
    ]

    for sess in sessions:
        print("\n" + "=" * 80)
        print(f"SESSION: {sess['session_name']}")
        print("=" * 80)

        # 1) compress raw Neuropixels files (.bin -> .cbin)
        compress_recordings(sess, keep_original=COMPRESS_KEEP_ORIGINAL)

        # 2) load AP recordings from compressed files
        load_recordings(sess)

        # 3) preprocessing
        preprocess_recordings(sess)

        # 4) spike sorting
        run_kilosort4(
            sess,
            params=KS_PARAMS,
            remove_existing_folder=KS_REMOVE_EXISTING_FOLDER,
        )

        # 5) build and save sorting analyzers
        build_sorting_analyzers(
            sess,
            n_jobs=ANALYZER_N_JOBS,
            chunk_duration=ANALYZER_CHUNK_DURATION,
        )

        # 6) export ALF / IBL GUI
        export_alf(
            sess,
            stop_on_error=False,
            n_jobs=EXPORT_N_JOBS,
            chunk_duration=EXPORT_CHUNK_DURATION,
        )


if __name__ == "__main__":
    main()