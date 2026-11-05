"""Evaluate the saved quadratic equation at a requested time."""

import argparse
import json
import math
from pathlib import Path


def predict(time, model_path="artifacts/model.json"):
    if not math.isfinite(time) or time < 0:
        raise ValueError("time must be finite and non-negative")
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    height = (model["intercept"] + model["coefficient_t"] * time
              + model["coefficient_t_squared"] * time**2)
    return {"time_seconds": time, "height_metres": height,
            "beyond_training_range": time > model["last_train_time_seconds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("time", type=float, help="Seconds since launch")
    parser.add_argument("--model", default="artifacts/model.json")
    args = parser.parse_args()
    try:
        print(json.dumps(predict(args.time, args.model), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Prediction failed: {error}\n")
