"""Frozen source contract for a future clean, original U-Sleep development fit.

This module never opens EDFs, labels, or model weights and does not authorize fit.
It describes the pinned U-Sleep recipe separately from U-Time and SLEEPYLAND.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .contracts import content_id, read_json
from .research import file_sha256


SLOT = "usleep_official"
REVISION = "7fc4cbf79e5454661f1c0d0768886e4dfbe9d42a"
SOURCE_MANIFEST_SHA256 = "59dd5d46898b57f8c148ece9e2518cff6c1a32f483969cdd6ffbd0552d23fa08"
SOURCE_FILES = (
    "resources/usleep_dataset_prep/sedf-sc.md",
    "resources/usleep_dataset_prep/sedf-st.md",
    "utime/bin/defaults/usleep/hparams.yaml",
    "utime/bin/defaults/usleep/dataset_configurations/sedf_sc.yaml",
    "utime/bin/defaults/usleep/dataset_configurations/sedf_st.yaml",
    "utime/models/usleep.py",
    "utime/bin/train.py",
    "utime/bin/extract.py",
    "utime/bin/evaluate.py",
    "utime/train/trainer.py",
    "utime/train/utils.py",
    "utime/sequences/balanced_random_batch_sequence.py",
    "utime/sequences/multi_sequence.py",
    "utime/sequences/batch_sequence.py",
    "utime/augmentation/augmenters.py",
    "utime/callbacks/callbacks.py",
    "utime/utils/scriptutils/train.py",
    "utime/bin/predict_one.py",
    "utime/bin/predict.py",
)
CHANNEL_GROUPS = (("EEG Fpz-Cz", "EOG horizontal"),
                  ("EEG Pz-Oz", "EOG horizontal"))
VIEW_NAMES = ("fpz_eog", "pz_eog")
LABEL_ORDER = ("W", "N1", "N2", "N3", "REM")
SAMPLES_PER_EPOCH = 30 * 128
CONTEXT = 35


def source_contract(root: Path) -> dict:
    """Verify pinned bytes, then return reviewed values from those exact bytes."""
    root = Path(root).resolve()
    manifest_path = root / "research/sources/utime.json"
    if file_sha256(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise ValueError("Original U-Sleep pinned source manifest changed")
    source = read_json(manifest_path)
    if (source.get("repository") != "perslev/U-Time" or
            source.get("source_url") != "https://github.com/perslev/U-Time/tree/" + REVISION or
            source.get("weights_downloaded") is not False or
            source.get("dataset_downloaded") is not False):
        raise ValueError("Original U-Sleep source or external artifact scope changed")
    files = {}
    for relative in SOURCE_FILES:
        evidence = source["files"].get(relative)
        if (type(evidence) is not dict or type(evidence.get("sha256")) is not str or
                file_sha256(root / "vendor/utime" / relative) != evidence["sha256"]):
            raise ValueError("Original U-Sleep required source/config bytes changed: " + relative)
        files[relative] = evidence["sha256"]
    lock = root / "requirements/utime.lock.txt"
    install = root / "research/runtimes/utime/install.json"
    if read_json(install).get("freeze_sha256") != file_sha256(lock):
        raise ValueError("Original U-Sleep isolated runtime lock/install differs")
    contract = {
        "schema_version": "1.0", "artifact_type": "original_usleep_source_contract",
        "slot": SLOT, "training_status": "NOT_READY_SOURCE_ONLY",
        "source_revision": REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_files_sha256": files,
        "runtime_lock_sha256": file_sha256(lock),
        "runtime_install_sha256": file_sha256(install),
        "adapter_sha256": file_sha256(Path(__file__)),
        "input": {"sample_rate_hz": 128, "epoch_seconds": 30,
                  "samples_per_epoch": SAMPLES_PER_EPOCH, "context_epochs": CONTEXT,
                  "channels_per_window": 2,
                  "channel_groups": [list(group) for group in CHANNEL_GROUPS],
                  "view_names": list(VIEW_NAMES),
                  "train_channel_sampling": "one_uniform_EEG_alternative_plus_EOG_per_lazy_queue_study_load",
                  "inference_views": "both_required_same_recording_and_original_grid",
                  "inference_aggregation": "sum_two_view_class_scores_then_argmax",
                  "inference_temporal_policy": "one_shot_per_true_continuous_block_no_tiling",
                  "tensor_shape": [64, CONTEXT, SAMPLES_PER_EPOCH, 2],
                  "signal_units_before_scaler": "volts",
                  "model_input_units": "recording_local_RobustScaler_dimensionless",
                  "label_order": list(LABEL_ORDER), "invalid_label": -1},
        "preprocessing": {"source_strip": "strip_to_match_label_guided",
                          "source_quality_control": "clip_noisy_values_zero_centered_global_20_IQR",
                          "source_scaler": "RobustScaler_per_recording_after_resample",
                          "clean_order": ["calibrated_volts", "continuous_resample_100_to_128_Hz",
                                          "cast_float32_like_native_H5", "channelwise_zero_centered_20_IQR_clip",
                                          "recording_local_RobustScaler", "final_float32"],
                          "batch_wise_scaling": False,
                          "clean_full_grid_strip": "disabled_requires_signal_only_timing",
                          "signal_breaks": "never_cross_true_continuous_block_boundaries",
                          "invalid_annotation": "retained_inside_continuous_signal_blocks",
                          "short_train_block": "nearest_boundary_epoch_pad_to_35_invalid_padding_targets"},
        "architecture": {"class": "USleep", "depth": 12, "kernel_size": 9,
                         "dilation": 1, "transition_window": 1,
                         "complexity_factor": 1.67, "activation": "elu",
                         "dense_classifier_activation": "tanh", "n_classes": 5,
                         "l2_reg": None, "init_filters": 5, "padding": "same",
                         "data_per_prediction": SAMPLES_PER_EPOCH},
        "fit": {"balanced_sampling": True, "sample_class_probabilities": [0.2] * 5,
                "margin_epochs": 17, "channel_mixture": False,
                "datasets": ["sedf_sc", "sedf_st"],
                "dataset_sample_alpha": 0.5,
                "dataset_probability": "0.5*fit_recording_count/cohort_total+0.25",
                "augmenters": [
                    {"class": "RegionalErase", "min_region_fraction": 0.001,
                     "max_region_fraction": 0.33, "log_sample": True, "apply_prob": 0.1},
                    {"class": "ChannelDropout", "drop_fraction": 0.5,
                     "apply_prob": 0.1}],
                "loss": "SparseCategoricalCrossentropy_NATIVE_NONE_ignore_out_of_bounds",
                "optimizer": {"source_class": "Adam", "runtime_class": "tf.keras.optimizers.legacy.Adam",
                              "learning_rate": 1e-7,
                              "amsgrad": True, "decay": 0.0,
                              "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-8},
                "batch_size": 64, "max_epochs": 12000,
                "cli_max_train_samples_per_epoch": 500000,
                "steps_per_epoch": "ceil(floor(min(T_fit,500000)/35)/64)"},
        "source_validation": {"monitor": "val_dice", "patience": 200,
                              "min_delta": 0, "mode": "max",
                              "per_dataset_max_studies_cli_default": 20,
                              "per_dataset_five_class_dice_rounded_decimals": 4,
                              "multi_dataset_aggregation": "unweighted_mean_of_dataset_scores",
                              "checkpoint": "weights_every_epoch_not_optimizer",
                              "restore_best_weights": False},
        "clean_selection": {"inner_split": "native_inner_v1_40_fit_8_validation",
                            "metric": "pooled_valid_original_grid_fixed5_macro_F1",
                            "arithmetic": "exact_fraction_macro_f1_to_float64_then_numpy_round4",
                            "cohort_aggregation": "none_pooled_epoch_confusion",
                            "validation_coverage": "all_inner_validation_nights",
                            "minimum_improvement": "strict_greater_min_delta_0",
                            "tie": "earliest_epoch", "patience": 200,
                            "max_epochs": 12000,
                            "outer_refit": "fresh_all48_selected_inner_epoch_count",
                            "outer12_role": "report_only"},
        "fold_guard_scope": "STRUCTURAL_ONLY_requires_external_frozen_protocol_and_record_inventory_equality",
        "clean_fit_unresolved": ["development_fold_steps_per_epoch",
                                 "native_batch64_resource_profile",
                                 "full_continuous_block_one_shot_resource_profile",
                                 "compiled_legacy_Adam_AMSGrad_numerical_and_resume_preflight",
                                 "signal_only_full_grid_preprocessing_parity",
                                 "short_signal_block_training_and_inference_parity"],
    }
    contract["contract_id"] = content_id(contract)
    return contract


def validate_development_fold(partition: dict, records: list[dict],
                              expected_development_participants: list[str]) -> None:
    """Require the caller's independently frozen original D participant roster."""
    if type(records) is not list or not records:
        raise ValueError("Original U-Sleep has no development records")
    ids = [row.get("recording_id") for row in records if type(row) is dict]
    participants = [row.get("participant_id") for row in records if type(row) is dict]
    if (len(ids) != len(records) or any(type(value) is not str or not value for value in ids) or
            len(set(ids)) != len(ids) or len(participants) != len(records) or
            any(type(row.get("n_epochs")) is not int or row["n_epochs"] < 1
                for row in records if type(row) is dict) or
            any(type(value) is not str or re.fullmatch(r"(?:SC|ST):[0-9]{2}", value) is None
                for value in participants) or
            any(not recording.startswith(participant[:2])
                for recording, participant in zip(ids, participants))):
        raise ValueError("Original U-Sleep D recording identity differs")
    universe = set(participants)
    if (type(expected_development_participants) is not list or
            len(expected_development_participants) != 60 or
            any(type(value) is not str or re.fullmatch(r"(?:SC|ST):[0-9]{2}", value) is None
                for value in expected_development_participants) or
            len(set(expected_development_participants)) != 60 or
            universe != set(expected_development_participants) or
            sum(value.startswith("SC:") for value in universe) != 46 or
            sum(value.startswith("ST:") for value in universe) != 14):
        raise ValueError("Original U-Sleep records differ from frozen D roster")
    groups = {}
    for key, count in (("outer_train", 48), ("outer_validation", 12),
                       ("inner_train", 40), ("inner_validation", 8)):
        values = partition.get(key) if type(partition) is dict else None
        if (type(values) is not list or len(values) != count or
                any(type(value) is not str for value in values) or
                len(set(values)) != count):
            raise ValueError("Original U-Sleep participant fold shape differs")
        groups[key] = set(values)
    if (len(universe) != 60 or groups["outer_train"] | groups["outer_validation"] != universe or
            groups["outer_train"] & groups["outer_validation"] or
            groups["inner_train"] | groups["inner_validation"] != groups["outer_train"] or
            groups["inner_train"] & groups["inner_validation"]):
        raise ValueError("Original U-Sleep participant leakage or D fold mismatch")
    outer_held = groups["outer_validation"]
    if (sum(value.startswith("SC:") for value in outer_held),
            sum(value.startswith("ST:") for value in outer_held)) not in ((9, 3), (10, 2)):
        raise ValueError("Original U-Sleep outer cohort allocation differs")
    inner_held = groups["inner_validation"]
    if (sum(value.startswith("SC:") for value in inner_held) != 6 or
            sum(value.startswith("ST:") for value in inner_held) != 2):
        raise ValueError("Original U-Sleep registered inner cohort allocation differs")


def training_dataset_probabilities(partition: dict, records: list[dict],
                                   expected_development_participants: list[str],
                                   phase: str) -> dict[str, float]:
    """Native MultiSequence alpha=.5 mixture over FIT recordings only."""
    return fit_schedule(partition, records, expected_development_participants,
                        phase)["sample_probabilities"]


def fit_schedule(partition: dict, records: list[dict],
                 expected_development_participants: list[str], phase: str) -> dict:
    """Exact per-fit counts and native epoch budget; invalid grid positions count."""
    validate_development_fold(partition, records, expected_development_participants)
    if phase not in ("inner", "outer"):
        raise ValueError("Original U-Sleep training phase differs")
    fit = set(partition["inner_train" if phase == "inner" else "outer_train"])
    counts = {cohort: sum(row["participant_id"] in fit and
                          row["participant_id"].startswith(cohort + ":")
                          for row in records) for cohort in ("SC", "ST")}
    total = sum(counts.values())
    if min(counts.values()) <= 0:
        raise ValueError("Original U-Sleep fit partition lacks a native training cohort")
    fit_records = [row for row in records if row["participant_id"] in fit]
    total_complete_epochs = sum(row["n_epochs"] for row in fit_records)
    capped = min(total_complete_epochs, 500000)
    contexts = capped // CONTEXT
    steps = (contexts + 63) // 64
    if steps < 1:
        raise ValueError("Original U-Sleep native fit has no update steps")
    return {"phase": phase, "recording_counts": counts,
            "sample_probabilities": {cohort: 0.5 * count / total + 0.25
                                     for cohort, count in counts.items()},
            "total_complete_epochs": total_complete_epochs,
            "context_samples_per_epoch": contexts, "steps_per_epoch": steps}


def short_train_block_padding(length_epochs: int) -> tuple[int, int]:
    """Registered boundary repetition counts for a true signal block below 35."""
    if type(length_epochs) is not int or length_epochs < 1:
        raise ValueError("Original U-Sleep signal block has no complete epoch")
    missing = max(CONTEXT - length_epochs, 0)
    return missing // 2, missing - missing // 2


def validate_fit_class_support(counts: dict[str, list[int]]) -> None:
    """Native balanced sampler cannot redistribute a cohort's missing class."""
    if (type(counts) is not dict or set(counts) != {"SC", "ST"} or
            any(type(values) is not list or len(values) != 5 or
                any(type(value) is not int or value <= 0 for value in values)
                for values in counts.values())):
        raise ValueError("Original U-Sleep fit cohort lacks a sampled class")


def _validate_signal_grid(signal: np.ndarray, epoch_index: np.ndarray,
                          channels: tuple[str, str], blocks: list[list[int]],
                          model_input_units: str) -> None:
    if (type(signal) is not np.ndarray or signal.dtype != np.float32 or
            signal.ndim != 3 or signal.shape[1:] != (SAMPLES_PER_EPOCH, 2) or
            len(signal) == 0 or not np.isfinite(signal).all() or
            type(epoch_index) is not np.ndarray or epoch_index.dtype.kind not in "iu" or
            not np.array_equal(epoch_index, np.arange(len(signal))) or
            channels not in CHANNEL_GROUPS or
            model_input_units != "recording_local_RobustScaler_dimensionless"):
        raise ValueError("Original U-Sleep full signal grid, order, declared units or channels differ")
    if (type(blocks) is not list or not blocks or
            any(type(item) is not list or len(item) != 2 or
                any(type(value) is not int for value in item) for item in blocks)):
        raise ValueError("Original U-Sleep continuous blocks are missing")
    cursor = 0
    for start, stop in blocks:
        if start != cursor or not start < stop <= len(signal):
            raise ValueError("Original U-Sleep continuous block order differs")
        cursor = stop
    if cursor != len(signal):
        raise ValueError("Original U-Sleep continuous blocks omit original epochs")


def validate_training_grid(signal: np.ndarray, labels: np.ndarray,
                           epoch_index: np.ndarray, channels: tuple[str, str],
                           blocks: list[list[int]], model_input_units: str) -> np.ndarray:
    _validate_signal_grid(signal, epoch_index, channels, blocks, model_input_units)
    if (type(labels) is not np.ndarray or labels.shape != (len(signal),) or
            labels.dtype.kind not in "iu" or np.any((labels < -1) | (labels > 4))):
        raise ValueError("Original U-Sleep labels must retain invalid original epochs")
    return labels >= 0


def validate_inference_grid(signal: np.ndarray, epoch_index: np.ndarray,
                            channels: tuple[str, str], blocks: list[list[int]],
                            model_input_units: str) -> None:
    """Signal-only input; the public inference signature has no reference labels."""
    _validate_signal_grid(signal, epoch_index, channels, blocks, model_input_units)


def aggregate_channel_views(view_scores: dict[str, np.ndarray],
                            epoch_indices: dict[str, np.ndarray]) -> np.ndarray:
    """Source predict_one policy: sum both views once, then first-index argmax."""
    if (type(view_scores) is not dict or set(view_scores) != set(VIEW_NAMES) or
            type(epoch_indices) is not dict or set(epoch_indices) != set(VIEW_NAMES)):
        raise ValueError("Original U-Sleep requires both named EEG+EOG views")
    arrays = [view_scores[name] for name in VIEW_NAMES]
    if (any(type(value) is not np.ndarray or value.dtype != np.float32 or
            value.ndim != 2 or value.shape[1] != 5 or len(value) == 0 or
            not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1)
            for value in arrays) or arrays[0].shape != arrays[1].shape):
        raise ValueError("Original U-Sleep view score grid or class order differs")
    if any(not np.allclose(value.sum(axis=1), 1.0, rtol=0, atol=1e-4)
           for value in arrays):
        raise ValueError("Original U-Sleep view scores are not class probabilities")
    if any(type(epoch_indices[name]) is not np.ndarray or
           epoch_indices[name].dtype.kind not in "iu" or
           not np.array_equal(epoch_indices[name], np.arange(len(arrays[0])))
           for name in VIEW_NAMES):
        raise ValueError("Original U-Sleep views do not share the original epoch grid")
    return np.argmax(arrays[0] + arrays[1], axis=1).astype(np.int8)
