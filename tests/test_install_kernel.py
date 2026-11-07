"""The kernel that install() uses when none is named: the Triton kernel on a device and the torch kernel on the CPU, and any named kernel as given."""

import pytest

pytest.importorskip("torch")


@pytest.mark.parametrize("device,expected", [("cuda", "triton"), ("cuda:0", "triton"), ("cpu", "torch"), ("hip", "triton")])
def test_the_default_kernel_of_a_device_is_triton_and_of_the_cpu_torch(device, expected):
    from velociraptor.triceratops import default_kernel

    assert default_kernel(device) == expected


def kernels_chosen(monkeypatch, *arguments, **options):
    from velociraptor import triceratops

    seen = []
    original = triceratops._bound

    def recording(function, device, kernel, exposure_nodes):
        seen.append(kernel)
        return original(function, device, kernel, exposure_nodes)

    monkeypatch.setattr(triceratops, "_bound", recording)
    with triceratops.install(*arguments, **options):
        pass
    return set(seen)


@pytest.mark.parametrize("device,expected", [("cuda", "triton"), ("cuda:0", "triton"), ("cpu", "torch")])
def test_install_without_a_kernel_a_device_runs_triton_and_the_cpu_runs_torch(monkeypatch, device, expected):
    pytest.importorskip("triceratops")
    assert kernels_chosen(monkeypatch, device) == {expected}


def test_a_named_kernel_is_used_as_given(monkeypatch):
    pytest.importorskip("triceratops")
    assert kernels_chosen(monkeypatch, "cuda", kernel="torch") == {"torch"}
    assert kernels_chosen(monkeypatch, "cuda", kernel="triton") == {"triton"}
