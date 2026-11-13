"""Characterization test for the real LeBel assembly.

Pins the numbers ``train_simple.py`` produces against a 2.8 GB fixture that is not
in the repository. Skips when the fixture is absent.

Running it::

    export LITCODER_LEBEL_ASSEMBLY=/path/to/assembly_lebel_uts03.pkl
    pytest tests/test_golden_lebel.py -v

Regenerating the golden values after an *intended* change::

    LITCODER_LEBEL_ASSEMBLY=/path/to/assembly_lebel_uts03.pkl \
        python tests/test_golden_lebel.py --update

Tolerances and the measurements behind them are in the JSON under
``tolerances.rationale`` and ``measured_variance``.
"""

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_lebel_uts03.json"
ENV_VAR = "LITCODER_LEBEL_ASSEMBLY"

# Same trimming train_simple.py uses for LeBel.
TRIMMING_CONFIG = {
    "train_features_start": 10,
    "train_features_end": -5,
    "train_targets_start": 0,
    "train_targets_end": None,
    "test_features_start": 50,
    "test_features_end": -5,
    "test_targets_start": 40,
    "test_targets_end": None,
}
FIR_DELAYS = [1, 2, 3, 4]


def load_golden() -> dict:
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def skip_reason(golden: dict) -> str:
    raw = os.environ.get(ENV_VAR)
    where = golden["fixture"]["source"]
    name = golden["fixture"]["file"]
    md5 = golden["fixture"]["md5"]
    if not raw:
        return (
            f"{ENV_VAR} is not set, so the real LeBel assembly is unavailable.\n"
            f"  This test is opt-in and CI is expected to skip it.\n"
            f"  To run it: export {ENV_VAR}=/path/to/{name}\n"
            f"  The file is ~2.8 GB (md5 {md5}); get it from: {where}."
        )
    return (
        f"{ENV_VAR} is set to {raw!r}, but no file exists there.\n"
        f"  Point it at {name} (~2.8 GB, md5 {md5}).\n"
        f"  Get it from: {where}."
    )


def file_md5(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def environment_fingerprint() -> dict:
    import scipy
    import torch

    if torch.backends.mps.is_available():
        device = "mps:0"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return {
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": device,
        "torch_threads": torch.get_num_threads(),
    }


def run_pipeline(assembly_path: str, results_dir: str) -> dict:
    """Run what train_simple.py runs, with logging and plotting stubbed.

    Calls ``fit_predict`` with no model kwargs, as ``train_simple.py`` calls
    ``trainer.train()`` with none -- the defaults are part of what is characterized.
    """
    from conftest import _NullBrainPlotter, _NullLogger
    import encoding.trainer as trainer_mod
    from encoding.assembly.assembly_loader import load_assembly
    from encoding.downsample.downsampling import Downsampler
    from encoding.features.factory import FeatureExtractorFactory
    from encoding.models.nested_cv import NestedCVModel

    assembly = load_assembly(assembly_path)
    extractor = FeatureExtractorFactory.create_extractor(
        modality="wordrate",
        model_name="wordrate",
        config={},
        cache_dir=str(Path(results_dir) / "cache"),
    )
    trainer = trainer_mod.AbstractTrainer(
        assembly=assembly,
        feature_extractors=[extractor],
        downsampler=Downsampler(),
        model=NestedCVModel(model_name="ridge_regression"),
        fir_delays=FIR_DELAYS,
        trimming_config=dict(TRIMMING_CONFIG),
        use_train_test_split=True,
        logger_backend="wandb",
        dataset_type="lebel",
        results_dir=results_dir,
        downsample_config={},
    )
    assert isinstance(trainer.experiment_logger, _NullLogger)
    assert isinstance(trainer.brain_plotter, _NullBrainPlotter)

    data = trainer.structure_data(
        trainer.apply_fir_delays(trainer.extract_and_downsample_features())
    )
    metrics, weights, alphas = trainer.model.fit_predict(
        features=data["Rstim"],
        targets=data["Rresp"],
        X_test=data["Pstim"],
        y_test=data["Presp"],
    )
    return {
        "shapes": {
            "X_train": list(data["Rstim"].shape),
            "Y_train": list(data["Rresp"].shape),
            "X_test": list(data["Pstim"].shape),
            "Y_test": list(data["Presp"].shape),
        },
        "correlations": np.asarray(metrics["correlations"]),
        "n_significant": int(metrics["n_significant"]),
        "median_score": float(metrics["median_score"]),
    }


@pytest.fixture(scope="module")
def golden() -> dict:
    return load_golden()


@pytest.fixture(scope="module")
def verified_assembly_path(golden) -> str:
    """Skip when the fixture is absent; fail when it is the wrong file."""
    raw = os.environ.get(ENV_VAR)
    if not raw or not Path(raw).is_file():
        pytest.skip(skip_reason(golden))

    path = Path(raw)
    actual = file_md5(path)
    expected = golden["fixture"]["md5"]
    if actual != expected:
        pytest.fail(
            f"Assembly at {path} has md5 {actual}; golden values were generated "
            f"against md5 {expected}.\n"
            f"If the new file is intentional, regenerate with:\n"
            f"  {golden['regeneration']['command']}"
        )
    return str(path)


@pytest.fixture(scope="module")
def result(verified_assembly_path, golden, tmp_path_factory) -> dict:
    """Module-scoped, so conftest's function-scoped autouse patching is not yet
    active; patch explicitly for the duration of the run."""
    with pytest.MonkeyPatch.context() as mp:
        from conftest import _NullBrainPlotter, _NullLogger
        import encoding.trainer as trainer_mod
        import wandb

        mp.setattr(trainer_mod, "WandBLogger", _NullLogger)
        mp.setattr(trainer_mod, "BrainPlotter", _NullBrainPlotter)
        mp.setattr(wandb, "init", lambda *a, **k: None, raising=False)

        results_dir = tmp_path_factory.mktemp("golden_results")
        return run_pipeline(verified_assembly_path, str(results_dir))


@pytest.fixture(scope="module")
def tolerances(golden) -> dict:
    """Pick the strict or loose set by fingerprint, and say which was used."""
    current = environment_fingerprint()
    recorded = golden["environment"]["fingerprint"]
    strict = current == recorded

    tol = golden["tolerances"]
    chosen = {
        "strict": strict,
        "quantile_atol": tol["quantile_atol_strict"] if strict else tol["quantile_atol_loose"],
        "count_rtol": tol["count_rtol_strict"] if strict else tol["count_rtol_loose"],
    }

    banner = (
        f"\n[golden_lebel] tolerance set: {'STRICT' if strict else 'LOOSE'} "
        f"(quantile atol={chosen['quantile_atol']:g}, count rtol={chosen['count_rtol']:g})"
    )
    if not strict:
        diff = {
            k: (recorded.get(k), current.get(k))
            for k in sorted(set(recorded) | set(current))
            if recorded.get(k) != current.get(k)
        }
        banner += (
            f"\n[golden_lebel] environment differs from the recorded baseline "
            f"(recorded -> current): {diff}"
        )
        warnings.warn(banner.strip(), stacklevel=2)
    print(banner)
    return chosen


def test_golden_json_is_well_formed(golden):
    for key in ("fixture", "shapes", "metrics", "quantiles", "tolerances", "regeneration"):
        assert key in golden
    assert golden["metrics"]["median_score"] == pytest.approx(golden["quantiles"]["50"])


def test_shapes_match_exactly(result, golden):
    assert result["shapes"] == {k: list(v) for k, v in golden["shapes"].items()}


def test_correlation_quantiles(result, golden, tolerances):
    correlations = result["correlations"]
    assert correlations.shape == (golden["metrics"]["n_voxels"],)

    atol = tolerances["quantile_atol"]
    failures = []
    for q_str, expected in sorted(golden["quantiles"].items(), key=lambda kv: float(kv[0])):
        actual = float(np.percentile(correlations, float(q_str)))
        if abs(actual - expected) > atol:
            failures.append(f"  p{q_str}: expected {expected:.10f}, got {actual:.10f} "
                            f"(delta {actual - expected:+.3e}, atol {atol:g})")
    assert not failures, (
        "Correlation quantiles moved:\n" + "\n".join(failures)
        + f"\nIf intended, regenerate with:\n  {golden['regeneration']['command']}"
    )


def test_n_significant_within_band(result, golden, tolerances):
    """Banded, not exact: voxels near the BH cutoff flip on tiny correlation changes."""
    expected = golden["metrics"]["n_significant"]
    actual = result["n_significant"]
    rtol = tolerances["count_rtol"]
    lo, hi = expected * (1 - rtol), expected * (1 + rtol)
    assert lo <= actual <= hi, (
        f"n_significant={actual} outside [{lo:.0f}, {hi:.0f}] (expected {expected}, rtol {rtol:g}).\n"
        f"If intended, regenerate with:\n  {golden['regeneration']['command']}"
    )


def test_threshold_counts_within_band(result, golden, tolerances):
    correlations = result["correlations"]
    rtol = tolerances["count_rtol"]
    for cut_str, expected in golden["threshold_counts"].items():
        actual = int((correlations > float(cut_str)).sum())
        lo, hi = expected * (1 - rtol), expected * (1 + rtol)
        assert lo <= actual <= hi, (
            f"voxels with r > {cut_str}: {actual} outside [{lo:.0f}, {hi:.0f}] "
            f"(expected {expected}, rtol {rtol:g}).\n"
            f"If intended, regenerate with:\n  {golden['regeneration']['command']}"
        )


def regenerate() -> int:
    """Rewrite the measured fields of the golden JSON; tolerances and notes are left alone."""
    import tempfile

    golden = load_golden()
    raw = os.environ.get(ENV_VAR)
    if not raw or not Path(raw).is_file():
        print(skip_reason(golden), file=sys.stderr)
        return 1

    actual_md5 = file_md5(Path(raw))
    if actual_md5 != golden["fixture"]["md5"]:
        print(
            f"note: fixture md5 changed ({golden['fixture']['md5']} -> {actual_md5}); "
            f"recording the new one.",
            file=sys.stderr,
        )

    import encoding.trainer as trainer_mod
    import wandb
    from conftest import _NullBrainPlotter, _NullLogger

    trainer_mod.WandBLogger = _NullLogger
    trainer_mod.BrainPlotter = _NullBrainPlotter
    wandb.init = lambda *a, **k: None

    with tempfile.TemporaryDirectory() as tmp:
        result = run_pipeline(raw, tmp)

    correlations = result["correlations"]
    golden["fixture"]["md5"] = actual_md5
    golden["run"]["date"] = datetime.date.today().isoformat()
    try:
        golden["run"]["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    golden["environment"]["fingerprint"] = environment_fingerprint()
    golden["shapes"] = result["shapes"]
    golden["metrics"]["median_score"] = round(float(np.median(correlations)), 10)
    golden["metrics"]["n_significant"] = result["n_significant"]
    golden["metrics"]["n_voxels"] = int(correlations.size)
    golden["quantiles"] = {
        q: round(float(np.percentile(correlations, float(q))), 10)
        for q in golden["quantiles"]
    }
    golden["threshold_counts"] = {
        cut: int((correlations > float(cut)).sum()) for cut in golden["threshold_counts"]
    }

    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")

    print(f"updated {GOLDEN_PATH}")
    print(f"  median_score  = {golden['metrics']['median_score']}")
    print(f"  n_significant = {golden['metrics']['n_significant']}")
    print("Review the diff before committing.")
    return 0


if __name__ == "__main__":
    if "--update" in sys.argv:
        sys.path.insert(0, str(Path(__file__).parent))
        sys.exit(regenerate())
    print(__doc__)
    print("Run the tests with pytest; pass --update to regenerate the golden values.")
    sys.exit(0)
