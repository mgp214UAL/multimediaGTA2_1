#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""InterCom buffering with optional adaptive acoustic feedback suppression."""

from collections import deque
import logging
import math
import os
import sys
import threading
import time

import numpy as np

import buffer
import minimal


class NLMS:
    """Normalized least-mean-squares adaptive FIR echo estimator.

    Uses Partitioned Block Frequency-Domain NLMS (PBFDAF) for real-time
    efficiency when the filter length and chunk size are >= 64 samples, and
    exact sample-by-sample time-domain NLMS for smaller sizes. Supports both
    mono (1D) and multi-channel (2D ``(frames, channels)``) chunks.
    """

    def __init__(self, L=4096, mu=0.5, eps=1e-6, channels=1, block_size=None):
        if L <= 0:
            raise ValueError("L must be greater than zero")
        if not 0 < mu <= 1:
            raise ValueError("mu must be in the range (0, 1]")
        if eps <= 0:
            raise ValueError("eps must be greater than zero")
        if channels <= 0:
            raise ValueError("channels must be greater than zero")
        if block_size is not None and block_size <= 0:
            raise ValueError("block_size must be greater than zero")

        self.L = int(L)
        self.mu = float(mu)
        self.eps = float(eps)
        self.channels = int(channels)
        self._requested_block_size = int(block_size) if block_size is not None else None
        self._lock = threading.Lock()
        self._init_state(self.channels, self._requested_block_size)

    def _init_state(self, channels, block_size):
        self.channels = int(channels)
        self._block_size = int(block_size) if block_size is not None else None
        self._use_fdaf = (
            self.L >= 64
            and self._block_size is not None
            and self._block_size >= 64
        )
        if self._use_fdaf:
            B = self._block_size
            self._P = (self.L + B - 1) // B
            self._M = 2 * B
            self._last_len = self.L - (self._P - 1) * B
            self._W = np.zeros((self.channels, self._P, B + 1), dtype=np.complex128)
            self._X_hist = np.zeros((self.channels, self._P, B + 1), dtype=np.complex128)
            self._x_prev = np.zeros((self.channels, B), dtype=np.float64)
            self._power = np.full((self.channels, B + 1), self.eps, dtype=np.float64)
        else:
            self._w_time = np.zeros((self.channels, self.L), dtype=np.float64)
            self._hist = np.zeros((self.channels, max(0, self.L - 1)), dtype=np.float64)

    def reset(self):
        """Reset learned filter coefficients and signal history."""
        with self._lock:
            self._init_state(self.channels, self._block_size)

    @property
    def w(self):
        """Return the current time-domain FIR filter coefficients."""
        with self._lock:
            if self._use_fdaf:
                taps = np.fft.irfft(self._W, n=self._M, axis=2)[:, :, :self._block_size]
                taps = taps.reshape(self.channels, -1)[:, :self.L]
            else:
                taps = self._w_time.copy()
        if self.channels == 1:
            return taps[0]
        return taps.T

    @staticmethod
    def _choose_block_size(frames):
        for candidate in (1024, 512, 256, 128, 64):
            if frames >= candidate and frames % candidate == 0:
                return candidate
        if frames >= 64:
            return min(frames, 1024)
        return frames

    @staticmethod
    def _to_2d(array):
        arr = np.asarray(array, dtype=np.float64)
        if arr.ndim == 1:
            return arr[:, np.newaxis]
        if arr.ndim == 2:
            return arr
        return arr.reshape(-1, arr.shape[-1])

    def _process_time_domain(self, x_2d, d_2d, adapt):
        frames, channels = d_2d.shape
        e_out = np.empty_like(d_2d)
        for ch in range(channels):
            if self.L > 1:
                buf = np.concatenate((self._hist[ch], x_2d[:, ch]))
            else:
                buf = x_2d[:, ch]
            w_ch = self._w_time[ch]
            for n in range(frames):
                u = buf[n:n + self.L][::-1]
                err = d_2d[n, ch] - float(w_ch @ u)
                e_out[n, ch] = err
                if adapt:
                    w_ch += (self.mu * err / (self.eps + float(u @ u))) * u
            if self.L > 1:
                self._hist[ch] = buf[-(self.L - 1):]
        return e_out

    def _process_fdaf(self, x_2d, d_2d, adapt):
        frames, channels = d_2d.shape
        B = self._block_size
        M = self._M
        e_out = np.empty_like(d_2d)
        # Scale step slightly to preserve stability across highly correlated signals.
        step_mu = 0.75 * self.mu

        for ch in range(channels):
            W = self._W[ch]
            X_hist = self._X_hist[ch]
            x_prev = self._x_prev[ch]
            power = self._power[ch]

            for off in range(0, frames, B):
                xb = x_2d[off:off + B, ch]
                db = d_2d[off:off + B, ch]

                x_win = np.concatenate((x_prev, xb))
                x_prev = xb

                X_new = np.fft.rfft(x_win)
                X_hist[1:] = X_hist[:-1]
                X_hist[0] = X_new

                Y = np.sum(W * X_hist, axis=0)
                y = np.fft.irfft(Y, n=M)[B:]
                eb = db - y
                e_out[off:off + B, ch] = eb

                if adapt:
                    inst_pow = np.sum(np.abs(X_hist) ** 2, axis=0)
                    power = 0.9 * power + 0.1 * inst_pow
                    reg = self.eps + 0.02 * float(np.mean(inst_pow))
                    denom = np.maximum(power, inst_pow) + reg

                    e_pad = np.zeros(M, dtype=np.float64)
                    e_pad[B:] = eb
                    E = np.fft.rfft(e_pad)

                    grad = np.conj(X_hist) * (E / denom)
                    g_time = np.fft.irfft(grad, n=M, axis=1)[:, :B]
                    if self._last_len < B:
                        g_time[-1, self._last_len:] = 0.0
                    g_pad = np.zeros((self._P, M), dtype=np.float64)
                    g_pad[:, :B] = g_time
                    W += step_mu * np.fft.rfft(g_pad, axis=1)

            self._x_prev[ch] = x_prev
            self._power[ch] = power

        return e_out

    def process(self, x, d, adapt=True):
        """Return the microphone signal after subtracting the estimated echo."""
        d_array = np.asarray(d)
        x_2d = self._to_2d(x)
        d_2d = self._to_2d(d_array)
        if x_2d.shape != d_2d.shape:
            raise ValueError("x and d must have the same shape")

        frames, channels = d_2d.shape
        with self._lock:
            if self._block_size is None or channels != self.channels:
                chosen_B = self._requested_block_size or self._choose_block_size(frames)
                self._init_state(channels, chosen_B)

            if self._use_fdaf and frames % self._block_size == 0:
                e_2d = self._process_fdaf(x_2d, d_2d, adapt)
            else:
                if self._use_fdaf:
                    # Fallback to time-domain if caller passes an unaligned length.
                    taps = np.fft.irfft(self._W, n=self._M, axis=2)[:, :, :self._block_size]
                    self._w_time = taps.reshape(self.channels, -1)[:, :self.L].copy()
                    self._hist = np.zeros((self.channels, max(0, self.L - 1)), dtype=np.float64)
                    self._use_fdaf = False
                e_2d = self._process_time_domain(x_2d, d_2d, adapt)

        e = e_2d.reshape(-1) if d_array.ndim == 1 else e_2d
        if np.issubdtype(d_array.dtype, np.integer):
            limits = np.iinfo(d_array.dtype)
            e = np.clip(np.rint(e), limits.min, limits.max)
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
    help="NLMS learning rate in (0, 1]",
)
minimal.parser.add_argument(
    "--nlms-length",
    type=int,
    default=4096,
    help="Number of taps per channel in the NLMS FIR filter",
)
minimal.parser.add_argument(
    "--double-talk-threshold",
    type=float,
    default=1.2,
    help="Microphone/reference RMS ratio above which NLMS adaptation is paused",
)
minimal.parser.add_argument(
    "--speaker-gain",
    type=float,
    default=1.0,
    help="Initial speaker playback gain in [0.0, 2.0]",
)
minimal.parser.add_argument(
    "--turn-taking",
    action="store_true",
    help="Automatically attenuate speaker output when near-end speech is detected",
)
minimal.parser.add_argument(
    "--turn-taking-attenuation",
    type=float,
    default=0.1,
    help="Speaker gain multiplier applied during near-end speech when --turn-taking is active",
)


class Feedback_supression(buffer.Buffering):
    """Buffering layer with optional NLMS processing and speaker turn-taking."""

    MIN_REF_RMS = 10.0
    NEAR_END_SPEECH_RMS = 150.0
    DTD_HANGOVER_CHUNKS = 3
    DUCK_HANGOVER_CHUNKS = 5

    def __init__(self):
        super().__init__()
        self._last_played_chunk = self.zero_chunk.copy()
        self._control_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._interactive = None

        self.suppression_enabled = bool(minimal.args.echo_suppression)
        self.double_talk_threshold = max(0.05, float(minimal.args.double_talk_threshold))
        self.speaker_gain = float(np.clip(minimal.args.speaker_gain, 0.0, 2.0))
        self.speaker_muted = False
        self.turn_taking = bool(minimal.args.turn_taking)
        self.turn_taking_attenuation = float(
            np.clip(minimal.args.turn_taking_attenuation, 0.0, 1.0)
        )

        history_chunks = max(
            1, math.ceil(minimal.args.nlms_length / minimal.args.frames_per_chunk)
        )
        self._ref_rms_history = deque([0.0] * history_chunks, maxlen=history_chunks)
        self._dtd_hangover = 0
        self._duck_hangover = 0
        self.double_talk_detected = False
        self.last_erle_db = 0.0

        self.nlms = None
        if self.suppression_enabled or self.turn_taking:
            if self.suppression_enabled:
                self.nlms = NLMS(
                    L=minimal.args.nlms_length,
                    mu=minimal.args.nlms_mu,
                    channels=minimal.args.number_of_channels,
                    block_size=minimal.args.frames_per_chunk,
                )
            self._interactive = threading.Thread(
                target=self._interactive_loop,
                daemon=True,
            )
            self._interactive.start()

    def _interactive_loop(self):
        print(
            "\n[Control Real-Time] '+'/'-': mu | '>'/'<': umbral DTD | "
            "']'/'[': vol. altavoz | 'm': mute | 't': turn-taking | "
            "'e': supresion on/off | 'r': reset filtro",
            flush=True,
        )
        if os.name == "nt":
            try:
                import msvcrt

                while not self._stop_event.is_set():
                    if not msvcrt.kbhit():
                        time.sleep(0.02)
                        continue
                    char = msvcrt.getwch()
                    self._handle_key(char)
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
                while not self._stop_event.is_set():
                    ready, _, _ = select.select([sys.stdin], [], [], 0.02)
                    if not ready:
                        continue
                    char = sys.stdin.read(1)
                    self._handle_key(char)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_key(self, char):
        if char in ("+", "-"):
            if self.nlms is None:
                return False
            with self.nlms._lock:
                if char == "+":
                    self.nlms.mu = min(1.0, self.nlms.mu + 0.05)
                    message = f"[Real-Time] mu aumentado a: {self.nlms.mu:.2f}"
                else:
                    self.nlms.mu = max(0.01, self.nlms.mu - 0.05)
                    message = f"[Real-Time] mu disminuido a: {self.nlms.mu:.2f}"
        elif char in (">", "<"):
            with self._control_lock:
                if char == ">":
                    self.double_talk_threshold = min(5.0, self.double_talk_threshold + 0.1)
                else:
                    self.double_talk_threshold = max(0.1, self.double_talk_threshold - 0.1)
                message = f"[Real-Time] umbral DTD: {self.double_talk_threshold:.2f}"
        elif char in ("]", "["):
            with self._control_lock:
                if char == "]":
                    self.speaker_gain = min(2.0, self.speaker_gain + 0.1)
                else:
                    self.speaker_gain = max(0.0, self.speaker_gain - 0.1)
                message = f"[Real-Time] ganancia altavoz: {self.speaker_gain:.2f}"
        elif char == "m":
            with self._control_lock:
                self.speaker_muted = not self.speaker_muted
                state = "SILENCIADO" if self.speaker_muted else "ACTIVO"
                message = f"[Real-Time] altavoz: {state}"
        elif char == "t":
            with self._control_lock:
                self.turn_taking = not self.turn_taking
                state = "ON" if self.turn_taking else "OFF"
                message = f"[Real-Time] turn-taking: {state}"
        elif char == "e":
            with self._control_lock:
                if self.nlms is None:
                    self.nlms = NLMS(
                        L=minimal.args.nlms_length,
                        mu=minimal.args.nlms_mu,
                        channels=minimal.args.number_of_channels,
                        block_size=minimal.args.frames_per_chunk,
                    )
                self.suppression_enabled = not self.suppression_enabled
                state = "ON" if self.suppression_enabled else "OFF"
                message = f"[Real-Time] supresion de eco: {state}"
        elif char == "r":
            if self.nlms is None:
                return False
            self.nlms.reset()
            message = "[Real-Time] filtro NLMS reiniciado"
        else:
            return False

        print(f"\n{message}", flush=True)
        return True

    @staticmethod
    def _rms(chunk):
        arr = np.asarray(chunk, dtype=np.float64)
        return float(np.sqrt(np.mean(arr * arr)))

    def _suppress_feedback(self, recorded_chunk):
        reference = self._last_played_chunk
        reference_rms = self._rms(reference)
        self._ref_rms_history.append(reference_rms)
        ref_window_rms = max(self._ref_rms_history)
        microphone_rms = self._rms(recorded_chunk)

        with self._control_lock:
            enabled = self.suppression_enabled
            threshold = self.double_talk_threshold

        near_end_loud = (
            ref_window_rms >= self.MIN_REF_RMS
            and microphone_rms > ref_window_rms * threshold
        )
        near_end_solo = (
            ref_window_rms < self.MIN_REF_RMS
            and microphone_rms > self.NEAR_END_SPEECH_RMS
        )

        if near_end_loud:
            self._dtd_hangover = self.DTD_HANGOVER_CHUNKS
        elif self._dtd_hangover > 0:
            self._dtd_hangover -= 1

        if near_end_loud or near_end_solo:
            self._duck_hangover = self.DUCK_HANGOVER_CHUNKS
        elif self._duck_hangover > 0:
            self._duck_hangover -= 1

        self.double_talk_detected = self._dtd_hangover > 0
        self._near_end_active = self._duck_hangover > 0

        if not enabled or self.nlms is None:
            self.last_erle_db = 0.0
            return recorded_chunk

        adapt = (ref_window_rms >= self.MIN_REF_RMS) and (not self.double_talk_detected)
        clean_chunk = self.nlms.process(reference, recorded_chunk, adapt=adapt)

        clean_rms = self._rms(clean_chunk)
        if microphone_rms > self.MIN_REF_RMS:
            self.last_erle_db = 20.0 * math.log10(
                microphone_rms / max(clean_rms, 1e-6)
            )
        else:
            self.last_erle_db = 0.0
        return clean_chunk

    def _apply_speaker_control(self, DAC):
        with self._control_lock:
            muted = self.speaker_muted
            gain = self.speaker_gain
            turn_taking = self.turn_taking
            attenuation = self.turn_taking_attenuation

        if muted:
            DAC[:] = 0
            return

        effective_gain = gain
        if turn_taking and getattr(self, "_near_end_active", False):
            effective_gain *= attenuation

        if abs(effective_gain - 1.0) > 1e-6:
            scaled = np.clip(
                np.rint(DAC.astype(np.float64) * effective_gain),
                -32768,
                32767,
            ).astype(DAC.dtype)
            DAC[:] = scaled

    def _is_active(self):
        return self.suppression_enabled or self.turn_taking or self.speaker_muted or abs(self.speaker_gain - 1.0) > 1e-6

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        if not self._is_active() and self.nlms is None:
            return super()._record_IO_and_play(ADC, DAC, frames, time, status)

        clean_chunk = self._suppress_feedback(ADC)
        super()._record_IO_and_play(clean_chunk, DAC, frames, time, status)
        self._apply_speaker_control(DAC)
        self._last_played_chunk = DAC.copy()

    def _read_IO_and_play(self, DAC, frames, time, status):
        if not self._is_active() and self.nlms is None:
            return super()._read_IO_and_play(DAC, frames, time, status)

        self.sent_chunk_number = (self.sent_chunk_number + 1) % self.CHUNK_NUMBERS
        read_chunk = self.read_chunk_from_file()
        clean_chunk = self._suppress_feedback(read_chunk)
        self.send(self.pack(self.sent_chunk_number, clean_chunk))
        chunk = self.unbuffer_next_chunk()
        self.play_chunk(DAC, chunk)
        self._apply_speaker_control(DAC)
        self._last_played_chunk = DAC.copy()
        return clean_chunk

    def run(self):
        try:
            super().run()
        finally:
            self._stop_event.set()


class Feedback_supression__verbose(Feedback_supression, buffer.Buffering__verbose):
    """Verbose Feedback_supression showing ERLE (dB) alongside buffer stats."""

    def __init__(self):
        super().__init__()
        self.erle_db_per_cycle = 0.0
        self.average_erle_db = 0.0
        self._erle_samples = []

    def _record_IO_and_play(self, ADC, DAC, frames, time, status):
        super()._record_IO_and_play(ADC, DAC, frames, time, status)
        if self.suppression_enabled:
            self._erle_samples.append(self.last_erle_db)
        if minimal.args.show_spectrum:
            self.recorded_chunk = ADC
            self.played_chunk = DAC

    def _read_IO_and_play(self, DAC, frames, time, status):
        chunk = super()._read_IO_and_play(DAC, frames, time, status)
        if self.suppression_enabled:
            self._erle_samples.append(self.last_erle_db)
        if minimal.args.show_spectrum:
            self.recorded_chunk = chunk
            self.played_chunk = DAC
        return chunk

    def cycle_feedback(self):
        if self._erle_samples:
            self.erle_db_per_cycle = sum(self._erle_samples) / len(self._erle_samples)
            self._erle_samples.clear()
        else:
            self.erle_db_per_cycle = 0.0
        self.average_erle_db = self.moving_average(
            self.average_erle_db, self.erle_db_per_cycle, self.cycle
        )
        super().cycle_feedback()

    def stats(self):
        return super().stats() + f"{self.erle_db_per_cycle:8.1f}"

    def first_line(self):
        return super().first_line() + f"{'ERLE':>8s}"

    def second_line(self):
        return super().second_line() + f"{'(dB)':>8s}"

    def separator(self):
        return super().separator() + "=" * 8

    def averages(self):
        return super().averages() + f"{self.average_erle_db:8.1f}"


try:
    import argcomplete
except ImportError:
    logging.warning("Unable to import argcomplete (optional)")


def main():
    """Run the feedback-suppression InterCom endpoint."""
    try:
        argcomplete.autocomplete(minimal.parser)
    except Exception:
        pass

    minimal.args = minimal.parser.parse_known_args()[0]
    if minimal.args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    if (
        minimal.args.show_stats
        or minimal.args.show_samples
        or minimal.args.show_spectrum
    ):
        intercom = Feedback_supression__verbose()
    else:
        intercom = Feedback_supression()

    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom._stop_event.set()
        intercom.print_final_averages()


if __name__ == "__main__":
    main()
