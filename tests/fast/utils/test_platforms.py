import pytest

from miles import platforms
from miles.platforms.sail import SailPlatform


def test_an_unknown_accelerator_is_refused(monkeypatch):
    """Any value but cuda, tpu or sail would otherwise run the CUDA platform silently."""
    monkeypatch.setenv("MILES_ACCELERATOR", "TPU")
    with pytest.raises(ValueError, match="expected cuda, tpu or sail"):
        platforms._select_platform()


def test_sail_is_probed_behind_the_cuda_default(monkeypatch):
    """SAIL runs on torch.cuda, so only the cuda-family probe can tell it apart from NVIDIA."""
    monkeypatch.delenv("MILES_ACCELERATOR", raising=False)
    monkeypatch.setattr(platforms, "_is_sail", lambda: True)
    platform = platforms._select_platform()
    assert isinstance(platform, SailPlatform)
    assert platform.device_type == "cuda"


def test_the_cuda_default_stays_cuda_without_the_sail_node(monkeypatch):
    monkeypatch.delenv("MILES_ACCELERATOR", raising=False)
    monkeypatch.setattr(platforms, "_is_sail", lambda: False)
    platform = platforms._select_platform()
    assert type(platform) is platforms.Platform


def test_an_explicit_sail_accelerator_needs_no_device_node(monkeypatch):
    monkeypatch.setenv("MILES_ACCELERATOR", "sail")
    monkeypatch.setattr(platforms, "_is_sail", lambda: False)
    assert isinstance(platforms._select_platform(), SailPlatform)


def test_is_sail_probes_the_alixpu_node(monkeypatch):
    monkeypatch.setattr(platforms.os.path, "exists", lambda p: p == "/dev/alixpu")
    assert platforms._is_sail() is True
    monkeypatch.setattr(platforms.os.path, "exists", lambda p: False)
    assert platforms._is_sail() is False


def test_sail_keeps_every_cuda_capability():
    """Only the reported identity moves: the NUMA bind, the allocator settings and the weight-push
    barrier all stay on, and torch.cuda remains the runtime."""
    platform = SailPlatform()
    assert platform.device_type == "cuda"
    assert platform.compile_backend == "inductor"
    assert platform.triton_kernels is True
    assert platform.flex_attention is True
    assert platform.rooted_reduce is True
    assert platform.premul_sum_reduce is True
    assert platform.weight_transfer_modes is None
    assert platform.prefill_weight_versions is True
    assert platform.dynamic_shapes is True
    assert platform.nvml_numa_affinity is True
    assert platform.rollout_kv_cache_dtype is None


def test_a_dynamic_helper_runs_eager_where_the_platform_compiles_static_shapes_only(monkeypatch):
    """The TPU compile backend rejects dynamic shapes, so the helper must stay the plain function there."""

    def helper(x):
        return x + 1

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", False)
    assert platforms.torch_compile(dynamic=True)(helper) is helper

    monkeypatch.setattr(platforms.current_platform, "dynamic_shapes", True)
    assert platforms.torch_compile(dynamic=True)(helper) is not helper
