import argparse
import numpy as np
import threading
import sys
import buffer
import minimal

class NLMS:
    """Implementación matemática del algoritmo NLMS (Sección 7.3)"""
    def __init__(self, L=2048, mu=0.5, eps=1e-6):
        self.L = L
        self.mu = mu
        self.eps = eps
        self.w = np.zeros(L)
        self.hist = np.zeros(L - 1)

    def process(self, x, d):
        x_float = x.astype(np.float32)
        d_float = d.astype(np.float32)
        buf = np.concatenate((self.hist, x_float))
        e = np.empty_like(d_float)
        for n in range(len(d_float)):
            u = buf[n:n + self.L][::-1]
            e[n] = d_float[n] - self.w @ u
            self.w += (self.mu * e[n] / (self.eps + u @ u)) * u
        self.hist = buf[-(self.L - 1):]
        return e.astype(d.dtype)

# =========================================================================
# SOLUCIÓN COMPLETA DE ARGUMENTOS: Reutilizamos el parser original del framework
# =========================================================================
# 1. Obtenemos el parser con todos los argumentos de minimal.py y buffer.py
parser = minimal.parser

# 2. Añadimos de manera limpia nuestros argumentos exigidos para el hito (9.1)
parser.add_argument('--echo-suppression', action='store_true', default=False,
                    help='Activa la supresión de eco acústico')
parser.add_argument('--nlms-mu', type=float, default=0.5,
                    help='Learning rate inicial para el algoritmo NLMS')

# 3. Parseamos y actualizamos la referencia global compartida que usan las clases padre
args = parser.parse_args()
minimal.args = args
buffer.args = args

# Detectamos la clase base disponible en tu archivo buffer.py
BaseClass = getattr(buffer, 'Buffering', getattr(buffer, 'Buffer', None))

class Feedback_supression(BaseClass):
    """Subclase oficial requerida por el entregable"""
    def __init__(self):
        # Inicializamos la clase padre (ahora tendrá acceso a todas sus variables de audio)
        super().__init__()

        self.last_played_chunk = None
        if args.echo_suppression:
            self.nlms = NLMS(L=2048, mu=args.nlms_mu)
            self.interactive = threading.Thread(target=self.interactive_loop, daemon=True)
            self.interactive.start()
        else:
            self.nlms = None

    def interactive_loop(self):
        print("\n[Control Real-Time] Introduce '+' para subir mu, '-' para bajar mu.", flush=True)
        while True:
            char = sys.stdin.read(1)
            if char == '+':
                self.nlms.mu = min(2.0, self.nlms.mu + 0.05)
                print(f"\n[Real-Time] mu aumentado a: {self.nlms.mu:.2f}", flush=True)
            elif char == '-':
                self.nlms.mu = max(0.01, self.nlms.mu - 0.05)
                print(f"\n[Real-Time] mu disminuido a: {self.nlms.mu:.2f}", flush=True)

    def record(self):
        recorded_chunk = super().record()
        if self.nlms is not None and self.last_played_chunk is not None:
            clean_chunk = self.nlms.process(self.last_played_chunk, recorded_chunk)
            return clean_chunk
        return recorded_chunk

    def play(self, chunk):
        self.last_played_chunk = chunk
        super().play(chunk)

if __name__ == "__main__":
    intercom = Feedback_supression()
    intercom.run()
