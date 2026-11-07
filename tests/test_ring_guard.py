"""The guard that stops a run at the first ring timeout of the kernel log, tested on a fake source of lines."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("ring_guard", Path(__file__).resolve().parent.parent / "benchmarks" / "ring_guard.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RingTimeout, ring_guard = module.RingTimeout, module.ring_guard


def test_a_ring_timeout_line_trips_the_guard_and_the_block_raises():
    calls = []
    with pytest.raises(RingTimeout, match="gfx_0.0.0 timeout"):
        with ring_guard(command=("printf", "quiet\\namdgpu 0000:03:00.0: ring gfx_0.0.0 timeout, signaled seq=1\\nafter\\n"), on_trip=lambda: calls.append(1)) as tripped:
            import time

            for _ in range(50):
                time.sleep(0.02)
    assert calls == [1]


def test_a_log_with_no_timeout_leaves_the_block_alone():
    with ring_guard(command=("printf", "ring gfx_0.0.0 reset succeeded\\nanything else\\n"), on_trip=lambda: pytest.fail("tripped")) as tripped:
        pass
    assert tripped == []


def test_a_trip_interrupts_the_main_thread_by_default():
    import threading

    never = threading.Event()  # never set: the main thread waits in steps of 0.1 s, and the guard's interrupt is raised when a wait returns. 60 s ends a test whose guard never fires
    with pytest.raises(RingTimeout):
        with ring_guard(command=("printf", "ring gfx_0.0.0 timeout\\n")):
            for _ in range(600):
                never.wait(0.1)
            pytest.fail("the guard did not interrupt the main thread")


def test_a_trip_that_arrives_while_the_guard_is_starting_still_raises_a_ring_timeout():
    """The watcher can read its line and interrupt the main thread before Thread.start() returns. That interrupt must still surface as RingTimeout and not as a bare KeyboardInterrupt."""
    import threading

    never = threading.Event()
    for _ in range(40):
        with pytest.raises(RingTimeout):
            with ring_guard(command=("printf", "ring gfx_0.0.0 timeout\\n")):
                for _ in range(600):
                    never.wait(0.01)
                pytest.fail("the guard did not interrupt the main thread")


def test_a_trip_interrupts_a_process_that_was_started_with_sigint_ignored_as_a_background_job_is():
    """interrupt_main() does nothing when SIGINT is ignored, as it is in a process started with `cmd &` from a non-interactive shell. The guard raises RingTimeout on leaving the block even when it did not interrupt it, so the test counts the waits inside the block."""
    import signal
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        f"""
        import importlib.util, signal, threading
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
        spec = importlib.util.spec_from_file_location("ring_guard", {str(Path(module.__file__))!r})
        guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
        never = threading.Event()
        waits = 0
        try:
            with guard.ring_guard(command=("printf", "ring gfx_0.0.0 timeout\\n")):
                for waits in range(1, 301):
                    never.wait(0.1)
        except guard.RingTimeout:
            pass
        print("WAITS", waits)
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN, "the earlier setting is back"
        """
    )
    done = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, timeout=60, preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
    waits = int(done.stdout.split("WAITS")[1].split()[0]) if "WAITS" in done.stdout else None
    assert done.returncode == 0 and waits is not None and waits < 300, f"the block ran to its end ({waits} of 300 waits): {done.stdout} {done.stderr}"
