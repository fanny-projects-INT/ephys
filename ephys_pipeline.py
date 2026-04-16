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

        compress_recordings(sess, keep_original=COMPRESS_KEEP_ORIGINAL)

        load_recordings(sess)

        preprocess_recordings(sess)

        run_kilosort4(
            sess,
            params=KS_PARAMS,
            remove_existing_folder=KS_REMOVE_EXISTING_FOLDER,
        )

        build_sorting_analyzers(
            sess,
            n_jobs=ANALYZER_N_JOBS,
            chunk_duration=ANALYZER_CHUNK_DURATION,
        )

        run_bombcell(sess)

        export_alf(
            sess,
            stop_on_error=False,
            n_jobs=EXPORT_N_JOBS,
            chunk_duration=EXPORT_CHUNK_DURATION,
        )


if __name__ == "__main__":
    main()