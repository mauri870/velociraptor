"""Stop a run on the first GPU ring timeout of the kernel log.

    from ring_guard import ring_guard
    with ring_guard():
        run_the_device_work()

Inside the block the kernel log is followed (`journalctl -k -f`); the first line that matches "ring .* timeout" interrupts the main thread (KeyboardInterrupt) and the block raises RingTimeout on leaving. Pass `command` for another source of lines and
`on_trip` for another action.

_thread.interrupt_main() does nothing when SIGINT is not handled by Python. A process started as a background job of a non-interactive shell (`cmd &`, nohup, a task runner) has SIGINT ignored. In the main thread the guard therefore
installs Python's handler for the block when SIGINT is ignored or the default, and puts the earlier setting back on leaving.

Limit: a Python signal handler runs between bytecodes of the main thread. The guard cannot stop a kernel in flight or a call that does not return. It stops the run once the running call returns to Python, so launch kernels in slices of about 10 ms."""
import _thread
import contextlib
import re
import signal
import subprocess
import threading

PATTERN = re.compile(r"ring .* timeout")


class RingTimeout(RuntimeError):
    pass


@contextlib.contextmanager
def ring_guard(command=("journalctl", "-k", "-f", "-n", "0", "--no-pager"), on_trip=_thread.interrupt_main):
    tripped = []
    process = subprocess.Popen(list(command), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def watch():
        for line in process.stdout:
            if PATTERN.search(line):
                tripped.append(line.strip())
                on_trip()
                return

    thread = threading.Thread(target=watch, daemon=True)
    earlier = None
    if threading.current_thread() is threading.main_thread() and signal.getsignal(signal.SIGINT) in (signal.SIG_IGN, signal.SIG_DFL):
        earlier = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        thread.start()  # a trip can arrive before start() returns; the interrupt is then raised here, inside the try
        yield tripped
    except KeyboardInterrupt:
        if tripped:
            raise RingTimeout(tripped[0]) from None
        raise
    finally:
        process.terminate()
        thread.join(timeout=2)
        if earlier is not None:
            signal.signal(signal.SIGINT, earlier)
    if tripped:
        raise RingTimeout(tripped[0])
