#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

"""Removes spatial (inter-channel) and temporal (intra-channel) redundancy using the overlapped MDCT (TDAC)."""

import numpy as np
import logging
import minimal
from stereo_MST_coding_32 import Stereo_MST_Coding_32 as Stereo_Coding
from stereo_MST_coding_32 import Stereo_MST_Coding_32__verbose as Stereo_Coding__verbose

minimal.parser.add_argument(
    "-w", "--window",
    type=str,
    default="sine",
    choices=["rectangular", "sine", "kbd"],
    help="Window function for MDCT (rectangular, sine, kbd) (default: sine)"
)

class Temporal_MDCT(Stereo_Coding):

    def __init__(self):
        super().__init__()
        logging.info(__doc__)

        self.N = minimal.args.frames_per_chunk
        self.window_name = minimal.args.window
        logging.info(f"MDCT window = {self.window_name}")

        n = np.arange(2 * self.N, dtype=np.float64)
        k = np.arange(self.N, dtype=np.float64)[:, None]
        self.mdct_matrix = np.cos((np.pi / self.N) * (n + 0.5 + self.N / 2.0) * (k + 0.5)) * np.sqrt(2.0 / self.N)

        if self.window_name == "rectangular":
            self.window = np.ones(2 * self.N, dtype=np.float64)
        elif self.window_name == "sine":
            self.window = np.sin((np.pi / (2.0 * self.N)) * (n + 0.5))
        elif self.window_name == "kbd":
            alpha = 4.0
            w_kaiser = np.kaiser(self.N + 1, np.pi * alpha)
            csum = np.cumsum(w_kaiser)
            w_half = np.sqrt(csum[:-1] / csum[-1])
            self.window = np.concatenate([w_half, w_half[::-1]])

        self.prev_analysis_chunk = np.zeros((self.N, minimal.args.number_of_channels), dtype=np.float64)
        self.prev_synthesis_half = np.zeros((self.N, minimal.args.number_of_channels), dtype=np.float64)

    def analyze_in_time(self, chunk):
        chunk_f = chunk.astype(np.float64)
        block_2N = np.vstack((self.prev_analysis_chunk, chunk_f))
        self.prev_analysis_chunk = chunk_f.copy()

        windowed = block_2N * self.window[:, None]
        coeffs = self.mdct_matrix @ windowed
        return np.rint(coeffs).astype(np.int32)

    def synthesize_in_time(self, coeffs):
        coeffs_f = coeffs.astype(np.float64)
        block_2N = (self.mdct_matrix.T @ coeffs_f) * self.window[:, None]

        reconstructed = self.prev_synthesis_half + block_2N[:self.N]
        self.prev_synthesis_half = block_2N[self.N:].copy()
        return np.rint(reconstructed).astype(np.int32)

    def analyze(self, chunk):
        chunk = Stereo_Coding.analyze(self, chunk)
        decomposition = self.analyze_in_time(chunk)
        return decomposition

    def synthesize(self, decomposition):
        chunk = self.synthesize_in_time(decomposition)
        chunk = Stereo_Coding.synthesize(self, chunk)
        return chunk


class Temporal_MDCT__verbose(Temporal_MDCT, Stereo_Coding__verbose):
    def __init__(self):
        super().__init__()
        self.original_chunks.append(self.zero_chunk)


try:
    import argcomplete
except ImportError:
    logging.warning("Unable to import argcomplete (optional)")

if __name__ == "__main__":
    minimal.parser.description = __doc__
    try:
        argcomplete.autocomplete(minimal.parser)
    except Exception:
        logging.warning("argcomplete not working :-/")
    minimal.args = minimal.parser.parse_known_args()[0]
    if minimal.args.show_stats or minimal.args.show_samples or minimal.args.show_spectrum:
        intercom = Temporal_MDCT__verbose()
    else:
        intercom = Temporal_MDCT()
    try:
        intercom.run()
    except KeyboardInterrupt:
        minimal.parser.exit("\nSIGINT received")
    finally:
        intercom.print_final_averages()
