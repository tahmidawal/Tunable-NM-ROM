"""Generate the authoritative, preregistered D0 promotion decision."""
from __future__ import annotations

import json
import sys

import numpy as np

input_path, output_path = sys.argv[1:3]
with open(input_path) as handle:
    d0 = json.load(handle)

config = d0["config"]
seed = int(config["data_seed"])
draw_count = int(config["draw_count"])
start, stop = config["selection_indices"]
stop += 1
rng = np.random.default_rng(seed)
cx = rng.uniform(0.15, 0.85, draw_count)
cy = rng.uniform(0.15, 0.85, draw_count)
distance = np.minimum.reduce((cx, cy, 1.0 - cx, 1.0 - cy))[start:stop]
nearest = np.array_split(np.argsort(distance), 4)[0]

concepts = {}
for name in ("HG4", "HG5"):
    concept = d0["concepts"][name]
    errors = np.asarray(concept["representation_oracle"]["trajectory_all"], np.float64)
    fraction = float(np.sum(np.square(errors[nearest])) / np.sum(np.square(errors)))
    concepts[name] = {
        "representation_oracle_mean": concept["representation_oracle"]["trajectory_mean"],
        "representation_oracle_worst": concept["representation_oracle"]["trajectory_worst"],
        "representation_gate_pass": bool(
            concept["representation_oracle"]["trajectory_mean"] <= 2e-4
            and concept["representation_oracle"]["trajectory_worst"] <= 7e-4
        ),
        "predictor_decoder_mean": concept["predictor_decoder"]["trajectory_mean"],
        "predictor_decoder_worst": concept["predictor_decoder"]["trajectory_worst"],
        "oracle_to_predictor_mean_ratio": concept[
            "oracle_to_predictor_trajectory_mean_ratio"
        ],
        "nearest_wall_quartile_source_indices": (nearest + start).tolist(),
        "nearest_wall_squared_representation_oracle_error_fraction": fraction,
        "wall_license_pass": bool(fraction >= 0.5),
        "basis_condition": concept["basis_gram_condition"],
    }

hg4_pass = concepts["HG4"]["representation_gate_pass"]
wall_fires = any(value["wall_license_pass"] for value in concepts.values())
decision = {
    "status": "authoritative_preregistered_decision",
    "d0_sha256_recorded_npz": d0["npz"]["sha256"],
    "selection_indices": [start, stop - 1],
    "nearest_wall_definition": "lowest parameter-to-wall-distance quartile",
    "non_authoritative_fields_ignored": [
        "concepts.*.wall_chart_licensed",
        "concepts.*.nearest_wall_squared_error_fraction",
    ],
    "concepts": concepts,
    "promotions": {
        "train_HG4": hg4_pass,
        "train_HG5": False,
        "train_HG5_reason": (
            "requires HG4 representation pass and a subsequent trained-HG4 decoder miss"
        ),
        "run_conditional_D1_HG5S": wall_fires,
    },
    "hard_stop_now": bool(not hg4_pass and not wall_fires),
}
with open(output_path, "w") as handle:
    json.dump(decision, handle, indent=1, allow_nan=False)
