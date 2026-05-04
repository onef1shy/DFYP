import json
from pathlib import Path

import fire
import torch

from config import DEFAULT_MODIS_OPERATOR_MAP, DEFAULT_SENTINEL_OPERATOR_MAP
from models import DFYPModel, SentinelDFYPRunner


def parse_year_operator_map(value):
    """Parse a year-to-operator mapping from dict, JSON, file, or CLI string."""
    if value is None:
        return dict(DEFAULT_MODIS_OPERATOR_MAP)
    if isinstance(value, dict):
        return {int(k): str(v) for k, v in value.items()}

    text = str(value).strip()
    if not text:
        return {}

    if Path(text).is_file():
        return parse_year_operator_map(json.loads(Path(text).read_text()))

    if text.startswith("{"):
        return parse_year_operator_map(json.loads(text))

    mapping = {}
    for item in text.split(","):
        year_text, operator_name = item.split(":", 1)
        mapping[int(year_text.strip())] = operator_name.strip()
    return mapping


def resolve_device(device):
    """Resolve a torch device from an optional CLI argument."""
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class RunTask:
    """Command-line entrypoints for DFYP training and evaluation."""

    @staticmethod
    def train_modis(
        cleaned_data_path=Path("datasets/modis/processed_data"),
        savedir=Path("checkpoints/modis"),
        dropout=0.5,
        dense_features=None,
        times="all",
        pred_years=None,
        num_runs=2,
        train_steps=25000,
        batch_size=64,
        starter_learning_rate=1e-4,
        weight_decay=1,
        l1_weight=0,
        patience=10,
        device=None,
        default_operator="learnable",
        year_operator_map=None,
    ):
        """Train the MODIS branch."""
        device = resolve_device(device)

        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"
        model = DFYPModel(
            in_channels=9,
            dropout=dropout,
            dense_features=dense_features,
            savedir=Path(savedir),
            device=device,
            default_operator=default_operator,
            year_operator_map=parse_year_operator_map(year_operator_map),
        )
        model.run(
            histogram_path,
            times,
            pred_years,
            num_runs,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )

    @staticmethod
    def eval_modis(
        checkpoint_path,
        cleaned_data_path=Path("datasets/modis/processed_data"),
        predict_year=2015,
        time=32,
        batch_size=64,
        dropout=0.5,
        dense_features=None,
        device=None,
        default_operator="learnable",
        year_operator_map=None,
    ):
        """Evaluate a MODIS checkpoint."""
        device = resolve_device(device)

        histogram_path = Path(cleaned_data_path) / "histogram_all_full.npz"
        model = DFYPModel(
            in_channels=9,
            dropout=dropout,
            dense_features=dense_features,
            savedir=Path("checkpoints/modis"),
            device=device,
            default_operator=default_operator,
            year_operator_map=parse_year_operator_map(year_operator_map),
        )
        return model.evaluate_checkpoint(
            checkpoint_path=Path(checkpoint_path),
            path_to_histogram=histogram_path,
            predict_year=predict_year,
            time=time,
            batch_size=batch_size,
        )

    @staticmethod
    def train_sentinel(
        root_dir=Path("datasets/sentinel2"),
        json_dir=Path("datasets/sentinel2/json"),
        savedir=Path("checkpoints/sentinel2"),
        crop=None,
        batch_size=64,
        train_steps=25000,
        starter_learning_rate=1e-4,
        weight_decay=1e-5,
        patience=10,
        device=None,
    ):
        """Train the Sentinel-2 branch."""
        device = resolve_device(device)

        crop_operator_map = dict(DEFAULT_SENTINEL_OPERATOR_MAP)
        crops = [crop.lower()] if crop is not None else ["corn", "cotton", "soybean", "winterwheat"]

        for crop_name in crops:
            operator_type = crop_operator_map[crop_name]
            runner = SentinelDFYPRunner(
                crop=crop_name,
                operator_type=operator_type,
                savedir=Path(savedir),
                device=device,
            )
            runner.run(
                root_dir=Path(root_dir),
                json_dir=Path(json_dir),
                train_steps=train_steps,
                batch_size=batch_size,
                starter_learning_rate=starter_learning_rate,
                weight_decay=weight_decay,
                patience=patience,
            )

    @staticmethod
    def eval_sentinel(
        checkpoint_path,
        root_dir=Path("datasets/sentinel2"),
        json_dir=Path("datasets/sentinel2/json"),
        crop="corn",
        device=None,
    ):
        """Evaluate a Sentinel-2 checkpoint."""
        device = resolve_device(device)

        crop_name = crop.lower()
        operator_type = dict(DEFAULT_SENTINEL_OPERATOR_MAP)[crop_name]
        runner = SentinelDFYPRunner(
            crop=crop_name,
            operator_type=operator_type,
            savedir=Path("checkpoints/sentinel2"),
            device=device,
        )
        return runner.evaluate_checkpoint(
            checkpoint_path=Path(checkpoint_path),
            root_dir=Path(root_dir),
            json_dir=Path(json_dir),
        )


if __name__ == "__main__":
    fire.Fire(RunTask)
