from pathlib import Path
import shutil
import spikeglx
from spikeinterface.extractors import read_cbin_ibl
import spikeinterface.preprocessing as spre
from spikeinterface.sorters import run_sorter, get_default_sorter_params
import spikeinterface.full as si
from spikeinterface.exporters import export_to_ibl_gui


def compress_recordings(sess, keep_original=True):
    """
    Compress Neuropixels .ap.bin and .lf.bin files for each probe.
    """
    session = sess["session_name"]
    probes = sess["probes"]

    for probe, P in probes.items():
        rec_folder = Path(P["rec_folder"])
        tag = f"{session} | {probe}"

        if not rec_folder.exists():
            print(f"[{tag}] rec_folder not found")
            continue

        print(f"[{tag}] compressing")

        bin_files = list(rec_folder.rglob("*.ap.bin")) + list(rec_folder.rglob("*.lf.bin"))

        if len(bin_files) == 0:
            print(f"[{tag}] no .bin files found")
            continue

        for f in bin_files:
            try:
                print(f"[{tag}] {f.name}")
                sr = spikeglx.Reader(f)
                sr.compress_file(keep_original=keep_original)
            except Exception as e:
                print(f"[{tag}] error on {f.name}: {e}")

    return sess


def build_paths(session_name: str, data_root: Path) -> dict:
    """
    Build all session paths and create output folders if needed.
    """
    mouse, date = session_name.split("_", 1)

    base_folder = data_root / mouse / date

    rec_root = base_folder / "Rec"
    ks_root = base_folder / "KS"
    alf_root = base_folder / "alf"
    analyzer_root = base_folder / "sorting_analyzer"
    bombcell_root = base_folder / "bombcell"
    shift_path = base_folder / "shift.txt"

    if not (rec_root / "probe00").is_dir():
        raise FileNotFoundError(f"[{session_name}] Missing folder: {rec_root / 'probe00'}")

    present_probes = ["probe00"] + (["probe01"] if (rec_root / "probe01").is_dir() else [])
    multi_probe = len(present_probes) > 1

    probes = {}
    for p in present_probes:
        probe_idx = int(p.replace("probe", ""))

        rec_folder = rec_root / p
        ks_folder = ks_root / p
        alf_folder = alf_root / p
        analyzer_folder = analyzer_root / p
        bombcell_folder = bombcell_root / p

        ks_folder.mkdir(parents=True, exist_ok=True)
        alf_folder.mkdir(parents=True, exist_ok=True)
        analyzer_folder.mkdir(parents=True, exist_ok=True)
        bombcell_folder.mkdir(parents=True, exist_ok=True)

        probes[p] = {
            "probe": p,
            "probe_idx": probe_idx,
            "rec_folder": rec_folder,
            "ks_folder": ks_folder,
            "alf_folder": alf_folder,
            "analyzer_folder": analyzer_folder,
            "bombcell_folder": bombcell_folder,
            "recording_json": ks_folder / "recording.json",
            "sorting_json": ks_folder / "sorting.json",
            "output_folder": ks_folder,
        }

    p0 = probes["probe00"]

    return {
        "session_name": session_name,
        "mouse": mouse,
        "date": date,
        "base_folder": base_folder,
        "rec_folder": p0["rec_folder"],
        "ks_folder": p0["ks_folder"],
        "alf_folder": p0["alf_folder"],
        "analyzer_folder": p0["analyzer_folder"],
        "bombcell_folder": p0["bombcell_folder"],
        "shift_path": shift_path,
        "recording_json": p0["recording_json"],
        "sorting_json": p0["sorting_json"],
        "output_folder": p0["output_folder"],
        "present_probes": present_probes,
        "multi_probe": multi_probe,
        "probes": probes,
    }


def load_recordings(sess: dict) -> dict:
    """
    Load AP recordings from compressed cbin files.
    """
    print(f"\nLoading: {sess['session_name']}")

    sess["recordings"] = {}

    for probe, P in sess["probes"].items():
        print(f"  {probe}")
        rec = read_cbin_ibl(
            P["rec_folder"],
            stream_name="ap",
        )
        sess["recordings"][probe] = rec

    return sess


def preprocess_recordings(sess):
    """
    Apply preprocessing to all recordings.
    """
    print(f"\nPreprocess: {sess['session_name']}")

    recordings = sess["recordings"]
    recordings_pp = {}
    preprocess_info = {}

    for probe_name, rec in recordings.items():
        print(f"  {probe_name}")

        rec_pp = spre.phase_shift(rec)

        rec_pp = spre.bandpass_filter(
            rec_pp,
            freq_min=300,
            freq_max=6000,
        )

        bad_channel_ids, channel_labels = spre.detect_bad_channels(
            rec_pp,
            method="coherence+psd",
        )

        print(f"    bad channels: {len(bad_channel_ids)}")

        if len(bad_channel_ids) > 0:
            rec_pp = spre.interpolate_bad_channels(
                rec_pp,
                bad_channel_ids=bad_channel_ids,
            )

        rec_pp = spre.common_reference(
            rec_pp,
            operator="median",
            reference="global",
        )

        recordings_pp[probe_name] = rec_pp
        preprocess_info[probe_name] = {
            "bad_channel_ids": list(bad_channel_ids),
            "channel_labels": list(channel_labels),
            "n_bad_channels": len(bad_channel_ids),
        }

    sess["recordings"] = recordings_pp
    sess["preprocess_info"] = preprocess_info

    return sess


def run_kilosort4(
    sess: dict,
    params: dict | None = None,
    remove_existing_folder: bool = True,
) -> dict:
    """
    Run Kilosort4 on all loaded probes and save JSON metadata.
    """
    if params is None:
        params = get_default_sorter_params("kilosort4")

    print(f"\nKilosort4: {sess['session_name']}")

    recordings = sess["recordings"]
    sess["sortings"] = {}

    for probe, recording in recordings.items():
        print(f"  {probe}")

        P = sess["probes"][probe]
        ks_folder = Path(P["ks_folder"])
        ks_folder.mkdir(parents=True, exist_ok=True)

        recording_json = ks_folder / "recording.json"
        sorting_json = ks_folder / "sorting.json"

        try:
            sorting = run_sorter(
                sorter_name="kilosort4",
                recording=recording,
                folder=ks_folder,
                remove_existing_folder=remove_existing_folder,
                verbose=True,
                **params,
            )

            recording.dump_to_json(recording_json, relative_to=ks_folder)
            sorting.dump_to_json(sorting_json, relative_to=ks_folder)

            sess["sortings"][probe] = sorting
            P["recording_json"] = recording_json
            P["sorting_json"] = sorting_json

            print(f"    done")

        except Exception as e:
            print(f"    failed: {e}")
            sess.setdefault("error_sorting", {})
            sess["error_sorting"][probe] = str(e)

    return sess


def build_sorting_analyzers(
    sess: dict,
    stop_on_error: bool = False,
    n_jobs: int = 1,
    chunk_duration: str = "1s",
    overwrite: bool = True,
) -> dict:
    """
    Build and compute a SortingAnalyzer for each probe.
    """
    session = sess["session_name"]
    probes = sess["probes"]
    recordings = sess["recordings"]
    sortings = sess["sortings"]

    analyzer_folders = {}

    print(f"\nAnalyzer: {session}")

    for probe, P in probes.items():
        tag = f"{session} | {probe}"

        if probe not in recordings:
            msg = f"[{tag}] missing recording"
            if stop_on_error:
                raise KeyError(msg)
            print(msg)
            continue

        if probe not in sortings:
            msg = f"[{tag}] missing sorting"
            if stop_on_error:
                raise KeyError(msg)
            print(msg)
            continue

        try:
            recording = recordings[probe]
            sorting = sortings[probe]
            analyzer_folder = Path(P["analyzer_folder"])
            analyzer_folder.parent.mkdir(parents=True, exist_ok=True)

            print(f"[{tag}] computing analyzer")

            analyzer = si.create_sorting_analyzer(
                sorting=sorting,
                recording=recording,
                format="binary_folder",
                folder=analyzer_folder,
                overwrite=overwrite,
            )

            analyzer.compute(
                [
                    "random_spikes",
                    "waveforms",
                    "templates",
                    "spike_amplitudes",
                    "spike_locations",
                    "noise_levels",
                    "quality_metrics",
                ],
                n_jobs=n_jobs,
                chunk_duration=chunk_duration,
                progress_bar=True,
            )

            analyzer_folders[probe] = analyzer_folder
            print(f"[{tag}] done")

        except Exception as e:
            msg = f"[{tag}] failed: {e}"
            if stop_on_error:
                raise
            print(msg)

    sess["analyzer_folders"] = analyzer_folders
    return sess


def export_alf(
    sess: dict,
    stop_on_error: bool = False,
    n_jobs: int = 1,
    chunk_duration: str = "1s",
    remove_existing: bool = True,
) -> dict:
    """
    Export each SortingAnalyzer to IBL GUI format.
    """
    session = sess["session_name"]
    probes = sess["probes"]
    analyzer_folders = sess["analyzer_folders"]

    print(f"\nExport ALF: {session}")

    for probe, P in probes.items():
        tag = f"{session} | {probe}"

        try:
            analyzer_folder = Path(analyzer_folders[probe])
            alf_folder = Path(P["alf_folder"])

            if not analyzer_folder.exists():
                raise FileNotFoundError(f"analyzer folder not found: {analyzer_folder}")

            if alf_folder.exists() and remove_existing:
                shutil.rmtree(alf_folder)

            alf_folder.mkdir(parents=True, exist_ok=True)

            print(f"[{tag}] exporting")

            analyzer = si.load_sorting_analyzer(analyzer_folder)

            export_to_ibl_gui(
                sorting_analyzer=analyzer,
                output_folder=alf_folder,
                lfp_recording=None,
                remove_if_exists=True,
                verbose=True,
                n_jobs=n_jobs,
                chunk_duration=chunk_duration,
                progress_bar=True,
            )

            print(f"[{tag}] done")

        except Exception as e:
            msg = f"[{tag}] failed: {e}"
            if stop_on_error:
                raise
            print(msg)

    return sess

def run_bombcell(
    sess: dict,
    stop_on_error: bool = False,
) -> dict:
    """
    Run Bombcell quality control on Kilosort outputs for each probe.
    Save results in a 'bombcell' folder at the session level.
    """

    import bombcell as bc

    session = sess["session_name"]
    probes = sess["probes"]

    print(f"\nBombcell: {session}")

    bombcell_outputs = {}

    for probe, P in probes.items():
        tag = f"{session} | {probe}"

        try:
            ks_folder = Path(P["ks_folder"])
            bombcell_folder = Path(sess["base_folder"]) / "bombcell" / probe

            bombcell_folder.mkdir(parents=True, exist_ok=True)

            print(f"[{tag}] running")

            param = bc.get_default_parameters(ks_folder)

            quality_metrics, param, unit_type, unit_type_string = bc.run_bombcell(
                ks_folder,
                bombcell_folder,
                param,
            )

            bombcell_outputs[probe] = bombcell_folder

            print(f"[{tag}] done")

        except Exception as e:
            msg = f"[{tag}] failed: {e}"
            if stop_on_error:
                raise
            print(msg)

    sess["bombcell_folders"] = bombcell_outputs

    return sess