import os
import subprocess
import sys
from pathlib import Path

SRC = Path.cwd().parent / "src"   # the notebook is in notebooks/, minimal.py is in src/

def run(script, *args):
    """Runs a script from src/, showing its output live.
    Interrupting the cell (■ button / Interrupt) kills the process and all its children."""
    cmd = [sys.executable, "-u", script, *args]   # -u: unbuffered output
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        cwd=SRC,
        stdin=subprocess.PIPE,                    # minimal.py expects input(); keep stdin open
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,                 # logging goes to stderr: merge it with stdout
        text=True,
        bufsize=1,
        creationflags=flags,
        start_new_session=(os.name != "nt"),
    )
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
    except KeyboardInterrupt:
        print("\n[Interrupted: stopping the process...]")
    finally:
        if proc.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        print(f"[Process finished, exit code {proc.returncode}]")
