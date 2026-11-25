#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""InterCom buffering with optional adaptive acoustic feedback suppression."""

import logging
import os
import sys
import time
import threading

import numpy as np

import buffer
import minimal


class NLMS:
    """Normalized least-mean-squares adaptive FIR echo estimator."""

    def __init__(self, L=2048, mu=0.5, eps=1e-6):
        if L <= 0:
            raise ValueError("L must be greater than zero")
        if not 0 < mu <= 1:
            raise ValueError("mu must be in the range (0, 1]")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")

        self.L = L
        self.mu = mu
        self.eps = eps
        self.w = np.zeros(L, dtype=np.float64)
        self.hist = np.zeros(L - 1, dtype=np.float64)
        self._lock = threading.Lock()

    def process(self, x, d, adapt=True):
        """Return the microphone signal after subtracting the estimated echo."""
        x_float = np.asarray(x, dtype=np.float64).reshape(-1)
        d_array = np.asarray(d)
        d_float = d_array.astype(np.float64, copy=False).reshape(-1)
        if x_float.size != d_float.size:
            raise ValueError("x and d must contain the same number of samples")

        buf = np.concatenate((self.hist, x_float))
        e = np.empty_like(d_float)
        with self._lock:
            for n in range(d_float.size):
                u = buf[n:n + self.L][::-1]
                e[n] = d_float[n] - self.w @ u
                if adapt:
                    self.w += (self.mu * e[n] / (self.eps + u @ u)) * u
            self.hist = buf[-(self.L - 1):]

        if np.issubdtype(d_array.dtype, np.integer):
            limits = np.iinfo(d_array.dtype)
            e = np.clip(e, limits.min, limits.max)
        return e.astype(d_array.dtype, copy=False).reshape(d_array.shape)


minimal.parser.add_argument(
    "--echo-suppression",
    "--eco-suppression",
    dest="echo_suppression",
    action="store_true",
    help="Enable adaptive NLMS acoustic feedback suppression (alias: --eco-suppression)",
)
minimal.parser.add_argument(
    "--nlms-mu",
    type=float,
    default=0.5,
    help="NLMS learning rate",
)
minimal.parser.add_argument(
    "--nlms-length",
    type=int,
    default=2048,
    help="Number of taps in the NLMS FIR filter",
)
minimal.parser.add_argument(
    "--double-talk-threshold",
    type=float,
    default=2.0,
    help="Microphone/reference RMS ratio above which NLMS adaptation is paused",
)


class Feedback_supression(buffer.Buffering):
    """Buffering layer with optional NLMS processing of microphone chunks."""

    def __init__(self):
        super().__init__()
        self._last_played_chunk = self.zero_chunk.copy()
        self.nlms = None
        if minimal.args.echo_suppression:
            self.nlms = NLMS(
                L=minimal.args.nlms_length,
                mu=minimal.args.nlms_mu,
            )
            self._interactive = threading.Thread(
                target=self._interactive_loop,
                daemon=True,
            )
            self._interactive.start()

    def _interactive_loop(self):
        print(
            "\n[Control Real-Time] Introduce '+' para subir mu, '-' para bajar mu.",
            flush=True,
        )
        if os.name == "nt":
            try:
                import msvcrt

                while self.nlms is not None:
                    if not msvcrt.kbhit():
                        time.sleep(0.02)
                        continue
                    char = msvcrt.getwch()
                    if not self._change_mu(char):
                        continue
                return
            except ImportError:
                pass

        if sys.stdin.isatty():
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self.nlms is not None:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.02)
                    if not ready:
                        continue
                    char = sys.stdin.read(1)
                    self._change_mu(char)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _change_mu(self, char):
        if char not in ("+", "-"):
            return False
        with self.nlms._lock:
            if char == "+":
                self.nlms.mu = min(1.0, self.nlms.mu + 0.05)
                message = f"[Real-Time] mu aumentado a: {self.nlms.mu:.2f}"
            else:
                self.nlms.mu = max(0.01, self.nlms.mu - 0.05)
                message = f"[Real-Time] mu disminuido a: {self.nlms.mu:.2f}"
        print(f"\n{message}", flush=True)
        return True

    def _suppress_feedback(self, recorded_chunk):
        if self.nlms is None:
            return recorded_chunk
        reference = self._last_played_chunk.reshape(-1)
        reference_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
        microphone_rms = float(np.sqrt(np.mean(recorded_chunk.astype(np.float64) ** 2)))
        threshold = minimal.args.double_talk_threshold
        adapt = microphone_rms <= max(1.0, reference_rms) * threshold
        return self.nlms.process(reference, recorded_chunk, adapt=adapt)

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        if self.nlms is None:
            return super()._record_IO_and_play(ADC, DAC, frames, time, status)

        self.sent_chunk_number = (self.sent_chunk_number + 1) % self.CHUNK_NUMBERS
        clean_chunk = self._suppress_feedback(ADC)
        self.send(self.pack(self.sent_chunk_number, clean_chunk))
        chunk = self.unbuffer_next_chunk()
        self._last_played_chunk = chunk.copy()
        self.play_chunk(DAC, chunk)

    def _read_IO_and_play(self, DAC, frames, time, status):
        if self.nlms is None:
            return super()._read_IO_and_play(DAC, frames, time, status)

        self.sent_chunk_number = (self.sent_chunk_number + 1) % self.CHUNK_NUMBERS
        read_chunk = self.read_chunk_from_file()
        clean_chunk = self._suppress_feedback(read_chunk)
        self.send(self.pack(self.sent_chunk_number, clean_chunk))
        chunk = self.unbuffer_next_chunk()
        self._last_played_chunk = chunk.copy()
        self.play_chunk(DAC, chunk)
        return clean_chunk


def main():
    """Run the feedback-suppression InterCom endpoint."""
    minimal.args = minimal.parser.parse_known_args()[0]
    if minimal.args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    intercom = Feedback_supression()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()


if __name__ == "__main__":
    main()
