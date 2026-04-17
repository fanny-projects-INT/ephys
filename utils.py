from pathlib import Path
import shutil
import json

import numpy as np
import pandas as pd
import spikeglx

from spikeinterface.extractors import read_cbin_ibl
import spikeinterface.preprocessing as spre
from spikeinterface.sorters import run_sorter, get_default_sorter_params
import spikeinterface.full as si
from spikeinterface.exporters import export_to_ibl_gui
import spikeinterface.curation as sc


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

            print("    done")

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
                    "unit_locations",
                    "noise_levels",
                    "template_metrics",
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
    Run Bombcell unit labeling from a SortingAnalyzer using SpikeInterface.
    Save labels for each probe in the bombcell folder.
    """
    session = sess["session_name"]
    probes = sess["probes"]

    print(f"\nBombcell: {session}")

    bombcell_outputs = {}

    for probe, P in probes.items():
        tag = f"{session} | {probe}"

        try:
            analyzer_folder = Path(P["analyzer_folder"])
            bombcell_folder = Path(P["bombcell_folder"])
            bombcell_folder.mkdir(parents=True, exist_ok=True)

            print(f"[{tag}] running")

            sorting_analyzer = si.load_sorting_analyzer(analyzer_folder)

            thresholds = sc.bombcell_get_default_thresholds()

            bombcell_labels = sc.bombcell_label_units(
                sorting_analyzer,
                thresholds=thresholds,
                label_non_somatic=True,
                split_non_somatic_good_mua=True,
            )

            bombcell_labels.to_csv(bombcell_folder / "bombcell_labels.csv")

            bombcell_outputs[probe] = {
                "folder": bombcell_folder,
                "labels": bombcell_labels,
            }

            print(f"[{tag}] done")

        except Exception as e:
            msg = f"[{tag}] failed: {e}"
            if stop_on_error:
                raise
            print(msg)

    sess["bombcell_outputs"] = bombcell_outputs
    return sess


def flatten_array_of_arrays(arr):
    out = []
    for x in arr:
        if isinstance(x, (list, np.ndarray)):
            out.extend(list(x))
        elif pd.notna(x):
            out.append(float(x))
    return np.array(out, dtype=float)


def detect_sync_events(
    sync,
    fs,
    threshold=32,
    win_s=1.0,
    min_ones_in_win=2,
    min_silence_s=1.0,
    hold_low_s=1.0,
):
    """
    Detect TTL-like sync events from the sync channel.
    Returns event times in seconds (ephys time base).
    """
    binary = (sync > threshold).astype(np.uint8)

    win_n = int(win_s * fs)
    kernel = np.ones(win_n, dtype=np.int32)
    count = np.convolve(binary, kernel, mode="same")
    active = (count >= min_ones_in_win).astype(np.uint8)

    one_idx = np.flatnonzero(binary)
    rise_times = []

    min_silence_n = int(min_silence_s * fs)
    hold_low_n = int(hold_low_s * fs)

    state = "inactive"
    last_end_idx = -999999

    i = 1
    n = len(sync)

    while i < n:
        if state == "inactive":
            if active[i - 1] == 0 and active[i] == 1 and (i - last_end_idx) >= min_silence_n:
                jpos = np.searchsorted(one_idx, i)
                if jpos < len(one_idx):
                    first_one = one_idx[jpos]
                    rise_times.append(first_one / fs)
                    state = "active"
                    i = max(i + 1, first_one + 1)
                    continue
        else:
            if active[i] == 0:
                low_run = 1
                k = i + 1
                while k < n and active[k] == 0 and low_run < hold_low_n:
                    low_run += 1
                    k += 1
                if low_run >= hold_low_n:
                    last_end_idx = k
                    state = "inactive"
                    i = k
                    continue
        i += 1

    return np.array(rise_times, dtype=float)


def correlate_full(x, y):
    return np.correlate(x, y, mode="full")


def fit_linear(x, y):
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def compute_and_save_alignment(
    sess: dict,
    db_path,
    mouse_behavior: str | None = None,
    lf_stream_name: str = "lp",
    sync_threshold: float = 32,
    coarse_bin_size: float = 0.1,
    fine_match_max_diff_s: float = 5.0,
    save_shift_txt: bool = True,
    save_affine_json: bool = True,
) -> dict:
    """
    Compute behavior/ephys alignment from LF sync channel and save:
    - shift.txt : intercept b only
    - alignment_affine.json : full affine transform t_behavior = a * t_ephys + b
    """
    session_name = sess["session_name"]
    mouse_ephys = sess["mouse"]
    date = sess["date"]
    base_folder = Path(sess["base_folder"])

    if mouse_behavior is None:
        mouse_behavior = mouse_ephys

    shift_path = Path(sess.get("shift_path", base_folder / "shift.txt"))
    affine_path = base_folder / "alignment_affine.json"

    print(f"\nAlignment: {session_name}")

    date_str = f"{date[:4]}-{date[5:7]}-{date[8:10]}"
    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Behavior DB not found: {db_path}")

    df = pd.read_feather(db_path)
    row = df[(df["Mouse_ID"] == mouse_behavior) & (df["Date"] == date_str)]

    if len(row) == 0:
        raise ValueError(f"No behavior entry found for {mouse_behavior} / {date_str}")

    row = row.iloc[0]

    bout_starts = np.asarray(row["Bout Start Times"], dtype=float)
    if len(bout_starts) == 0:
        raise ValueError("No bout start times found in behavior DB")

    lick_rewarded = flatten_array_of_arrays(row["Times Rewarded Licks"])
    lick_nonrewarded = flatten_array_of_arrays(row["Times Non Rewarded Licks"])
    lick_invalid = flatten_array_of_arrays(row["Times Invalid Licks"])

    lick_arrays = [lick_rewarded, lick_nonrewarded, lick_invalid]
    lick_arrays = [x for x in lick_arrays if len(x) > 0]
    if len(lick_arrays) > 0:
        lick_times = np.sort(np.concatenate(lick_arrays))
    else:
        lick_times = np.array([], dtype=float)

    sess["behavior"] = {
        "bout_starts": bout_starts,
        "lick_rewarded": lick_rewarded,
        "lick_nonrewarded": lick_nonrewarded,
        "lick_invalid": lick_invalid,
        "lick_times": lick_times,
        "mouse_behavior": mouse_behavior,
        "date_str": date_str,
    }

    print(f"  behavior bouts: {len(bout_starts)}")

    alignment_results = {}

    for probe, P in sess["probes"].items():
        tag = f"{session_name} | {probe}"
        rec_folder = Path(P["rec_folder"])

        lf_candidates = sorted(rec_folder.glob("*.lf.cbin"))
        ap_candidates = sorted(rec_folder.glob("*.ap.cbin"))

        if len(lf_candidates) == 0:
            raise FileNotFoundError(f"[{tag}] No LF cbin file found in {rec_folder}")
        if len(ap_candidates) == 0:
            raise FileNotFoundError(f"[{tag}] No AP cbin file found in {rec_folder}")

        lf_cbin_path = lf_candidates[0]
        ap_cbin_path = ap_candidates[0]

        print(f"[{tag}] loading LF sync from {lf_cbin_path.name}")

        rec_lf = read_cbin_ibl(
            folder_path=rec_folder,
            cbin_file_path=lf_cbin_path,
            load_sync_channel=True,
            stream_name=lf_stream_name,
        )

        fs_lf = rec_lf.get_sampling_frequency()
        sync = rec_lf.get_traces(
            start_frame=0,
            end_frame=rec_lf.get_num_frames(),
        )[:, -1]

        ts_events = detect_sync_events(sync, fs_lf, threshold=sync_threshold)

        print(f"[{tag}] sync events detected: {len(ts_events)}")

        if len(ts_events) == 0:
            raise ValueError(f"[{tag}] No sync events detected on LF sync channel")

        tmax = int(np.ceil(max(bout_starts[-1], ts_events[-1]) / coarse_bin_size)) + 1

        v_beh = np.zeros(tmax, dtype=float)
        v_eph = np.zeros(tmax, dtype=float)

        beh_idx = (bout_starts / coarse_bin_size).astype(int)
        eph_idx = (ts_events / coarse_bin_size).astype(int)

        beh_idx = beh_idx[(beh_idx >= 0) & (beh_idx < tmax)]
        eph_idx = eph_idx[(eph_idx >= 0) & (eph_idx < tmax)]

        v_beh[beh_idx] = 1
        v_eph[eph_idx] = 1

        xcorr = correlate_full(v_beh, v_eph)
        lags = np.arange(-len(v_eph) + 1, len(v_beh))
        best_lag = lags[np.argmax(xcorr)] * coarse_bin_size

        ts_events_shifted = ts_events + best_lag

        paired_x = []
        paired_y = []

        for y in bout_starts:
            diffs = np.abs(ts_events_shifted - y)
            idx = np.argmin(diffs)
            if diffs[idx] < fine_match_max_diff_s:
                paired_x.append(ts_events[idx])
                paired_y.append(y)

        paired_x = np.array(paired_x, dtype=float)
        paired_y = np.array(paired_y, dtype=float)

        if len(paired_x) < 3:
            raise ValueError(f"[{tag}] Not enough matched events for fine alignment")

        a, b = fit_linear(paired_x, paired_y)

        print(f"[{tag}] coarse shift estimate: {best_lag:.3f} s")
        print(f"[{tag}] fine alignment: t_behavior = {a:.8f} * t_ephys + {b:.8f}")
        print(f"[{tag}] matched events: {len(paired_x)}")

        probe_shift_path = base_folder / f"{probe}_shift.txt"
        probe_affine_path = base_folder / f"{probe}_alignment_affine.json"

        if save_shift_txt:
            probe_shift_path.write_text(f"{b:.8f}\n", encoding="utf-8")

        if save_affine_json:
            with open(probe_affine_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session_name": session_name,
                        "mouse": mouse_ephys,
                        "mouse_behavior": mouse_behavior,
                        "date": date,
                        "probe": probe,
                        "a": a,
                        "b": b,
                        "n_matched_events": int(len(paired_x)),
                        "coarse_shift_s": float(best_lag),
                        "lf_cbin_file": str(lf_cbin_path),
                        "ap_cbin_file": str(ap_cbin_path),
                        "fs_lf": float(fs_lf),
                    },
                    f,
                    indent=2,
                )

        alignment_results[probe] = {
            "a": a,
            "b": b,
            "n_matched_events": int(len(paired_x)),
            "coarse_shift_s": float(best_lag),
            "lf_cbin_file": str(lf_cbin_path),
            "ap_cbin_file": str(ap_cbin_path),
            "probe_shift_path": str(probe_shift_path),
            "probe_affine_path": str(probe_affine_path),
        }

    if len(alignment_results) == 1:
        only_probe = next(iter(alignment_results))
        res = alignment_results[only_probe]

        if save_shift_txt:
            shift_path.write_text(f"{res['b']:.8f}\n", encoding="utf-8")

        if save_affine_json:
            with open(affine_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session_name": session_name,
                        "mouse": mouse_ephys,
                        "mouse_behavior": mouse_behavior,
                        "date": date,
                        "probe": only_probe,
                        "a": res["a"],
                        "b": res["b"],
                        "n_matched_events": res["n_matched_events"],
                        "coarse_shift_s": res["coarse_shift_s"],
                        "lf_cbin_file": res["lf_cbin_file"],
                        "ap_cbin_file": res["ap_cbin_file"],
                    },
                    f,
                    indent=2,
                )

    sess["alignment"] = alignment_results
    return sess