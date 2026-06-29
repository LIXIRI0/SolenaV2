import os

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from config import (
    BATCH_SIZE,
    DISTRIBUTED_INIT_TIMEOUT,
    NUM_DEVICES,
    PER_DEVICE_BATCH_SIZE,
    SEQ_LEN,
    USE_MESH,
)


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return int(value)


def _local_device_ids():
    value = os.getenv("JAX_LOCAL_DEVICE_IDS")
    if value is None or value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def initialize_distributed() -> None:
    if not USE_MESH or jax.distributed.is_initialized():
        return

    coordinator = os.getenv("JAX_COORDINATOR_ADDRESS")
    num_processes = _env_int("JAX_NUM_PROCESSES")
    process_id = _env_int("JAX_PROCESS_ID")
    manual_env_values = (coordinator, num_processes, process_id)
    if any(value is not None for value in manual_env_values) and not all(
        value is not None for value in manual_env_values
    ):
        raise RuntimeError(
            "set all distributed launch env vars or none of them: "
            "JAX_COORDINATOR_ADDRESS, JAX_NUM_PROCESSES, JAX_PROCESS_ID"
        )

    kwargs = {
        "initialization_timeout": DISTRIBUTED_INIT_TIMEOUT,
    }
    if coordinator is not None and num_processes is not None and process_id is not None:
        kwargs.update(
            {
                "coordinator_address": coordinator,
                "num_processes": num_processes,
                "process_id": process_id,
                "local_device_ids": _local_device_ids(),
            }
        )

    jax.distributed.initialize(**kwargs)


def process_info() -> str:
    return (
        f"process={jax.process_index()}/{jax.process_count()} | "
        f"local_devices={jax.local_device_count()} | global_devices={jax.device_count()}"
    )


def is_primary_process() -> bool:
    return jax.process_index() == 0


def print_once(*args, **kwargs) -> None:
    if is_primary_process():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def create_data_mesh() -> Mesh:
    devices = np.asarray(jax.devices())
    if devices.size != NUM_DEVICES:
        raise RuntimeError(
            f"expected {NUM_DEVICES} global devices for mesh, got {devices.size}; "
            f"{process_info()}"
        )
    return Mesh(devices, ("data",))


def batch_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P("data", None))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def local_batch_size() -> int:
    return jax.local_device_count() * PER_DEVICE_BATCH_SIZE


def global_batch_shape() -> tuple[int, int]:
    return (BATCH_SIZE, SEQ_LEN)


def make_global_batch(local_data: np.ndarray, sharding: NamedSharding) -> jax.Array:
    expected_local_batch = local_batch_size()
    if local_data.shape[0] != expected_local_batch:
        raise ValueError(
            f"local batch has first dimension {local_data.shape[0]}, "
            f"expected {expected_local_batch}"
        )
    return jax.make_array_from_process_local_data(
        sharding,
        local_data,
        global_shape=global_batch_shape(),
    )
