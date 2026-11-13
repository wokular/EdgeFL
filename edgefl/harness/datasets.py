"""Dataset profiles: the data-handler + DB + table group for an FL task.

Fixed per dataset (not per node, not per run). A config names a dataset; the renderer
pulls the matching profile. Mirrors the committed env files (e.g. mnist1.env).

Under 1:1:1 each node has its own Postgres DB, so train/test table names stay constant
across nodes (same name, separate DBs = isolated). The shared-operator fallback would
need per-node table names; left as a future hook.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetProfile:
    name: str               # registry key, e.g. "mnist"
    module_name: str        # MODULE_NAME, data handler class
    module_file: str        # MODULE_FILE, file under TRAINING_APPLICATION_DIR
    logical_database: str   # LOGICAL_DATABASE (also the operators' DEFAULT_DBMS)
    train_table: str        # TRAIN_TABLE
    test_table: str         # TEST_TABLE
    extra_env: tuple = ()   # ((KEY, value), ...) extra node env the handler reads
    seed_rounds: int = 20   # rounds of data `seed` inserts when not told otherwise
    # MNIST rows carry the whole image as long text, which AnyLog's sqlite path turns into
    # an unquoted .blob token that the INSERT can't parse; it needs Postgres operators.
    requires_psql: bool = False


DATASETS: dict[str, DatasetProfile] = {
    "mnist": DatasetProfile(
        name="mnist",
        module_name="MnistDataHandler",
        module_file="custom_data_handler.py",
        logical_database="mnist_fl",
        train_table="mnist_train",
        test_table="mnist_test",
        requires_psql=True,
    ),
    # BloodMNIST. The publisher inserts scalar rows (filename, label, round_number) and
    # leaves the PNGs on disk, so sqlite works. Values mirror env_files/medmnist/.
    "medmnist": DatasetProfile(
        name="medmnist",
        module_name="MedMNISTDataHandler",
        module_file="medmnist_data_handler.py",
        logical_database="mydb",
        train_table="medmnist_train",
        test_table="medmnist_test",
        seed_rounds=10,
        extra_env=(
            ("IMAGE_ROOT_DIR", "edgefl/data/medmnist/published_images"),
            ("QUERY_RETRIES", "15"),
            ("QUERY_RETRY_SECONDS", "2"),
            ("TRAIN_EPOCHS", "20"),
            ("TRAIN_HISTORY_ROUNDS", "4"),
            ("EVAL_HISTORY_ROUNDS", "4"),
            ("BATCH_SIZE", "32"),
            ("LEARNING_RATE", "0.001"),
        ),
    ),
}


def get_dataset(name: str) -> DatasetProfile:
    if name not in DATASETS:
        raise KeyError(
            f"unknown dataset '{name}'. Known: {sorted(DATASETS)}. "
            f"Add a DatasetProfile to harness/datasets.py."
        )
    return DATASETS[name]
