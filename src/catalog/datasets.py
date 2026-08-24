

from dataclasses import dataclass
from typing import Callable

from loaders.tuab_loader import load_all_data as load_tuab_data
from loaders.vepiset_loader import load_all_data as load_vepiset_data
from runtime import default_data_dir


@dataclass(frozen=True)
class DatasetAdapter:
    name: str
    load_all_data: Callable
    default_data_dir: str
    default_train_subs: str
    default_val_subs: str
    default_test_subs: str
    sfreq: int


DATASET_REGISTRY = {
    "tuab": DatasetAdapter(
        name="tuab",
        load_all_data=load_tuab_data,
        default_data_dir=default_data_dir("TUAB"),
        default_train_subs="official",
        default_val_subs="official",
        default_test_subs="official",
        sfreq=250,
    ),
    "vepiset": DatasetAdapter(
        name="vepiset",
        load_all_data=load_vepiset_data,
        default_data_dir=default_data_dir("vepiset"),
        default_train_subs="auto",
        default_val_subs="auto",
        default_test_subs="auto",
        sfreq=250,
    ),
}


def list_datasets():
    return DATASET_REGISTRY.keys()


def get_dataset_adapter(name: str) -> DatasetAdapter:
    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}; available: {list(DATASET_REGISTRY)}") from exc
