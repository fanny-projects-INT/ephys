from pathlib import Path
from config import (
    DATA_ROOT,
    KS_PARAMS,
    COMPRESS_KEEP_ORIGINAL,
    KS_REMOVE_EXISTING_FOLDER,
    ANALYZER_N_JOBS,
    ANALYZER_CHUNK_DURATION,
    EXPORT_N_JOBS,
    EXPORT_CHUNK_DURATION,
)
from utils import (
    build_paths,
    compress_recordings,
    load_recordings,
    preprocess_recordings,
    run_kilosort4,
    build_sorting_analyzers,
    run_bombcell,
    export_alf,
)

# =========================================================
# SESSSIONS TO PROCESS
# =========================================================

SESSION_LIST = [
    "VF074v3_2026_03_24",
]

# =========================================================
# MAIN
# =========================================================
def main():
    sessions = [
        build_paths(session_name, data_root=DATA_ROOT)
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

        # 6) Bombcell QC
        run_bombcell(sess)

        # 7) export ALF / IBL GUI
        export_alf(
            sess,
            stop_on_error=False,
            n_jobs=EXPORT_N_JOBS,
            chunk_duration=EXPORT_CHUNK_DURATION,
        )


if __name__ == "__main__":
    main()