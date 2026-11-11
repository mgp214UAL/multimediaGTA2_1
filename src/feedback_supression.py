import argparse
import numpy as np
import threading
import sys
import buffer
import minimal

class NLMS:
    """Implementación matemática del algoritmo NLMS (Sección 7.3)"""
    def __init__(self, L=1024, mu=0.5, eps=1e-6):
        self.L = L
        self.mu = mu
        self.eps = eps
        self.w = np.zeros(L)
        self.hist = np.zeros(L - 1)

    def process(self, x, d):
        # Aseguramos que el tamaño de las señales se adapte dinámicamente a L
        if len(x) != self.L:
            self.L = len(x)
            self.w = np.zeros(self.L)
            self.hist = np.zeros(self.L - 1)

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

# Reutilizamos el parser original de la jerarquía de intercomunicación
parser = minimal.parser
parser.add_argument('--echo-suppression', action='store_true', default=False)
parser.add_argument('--nlms-mu', type=float, default=0.5)

args = parser.parse_args()
minimal.args = args
buffer.args = args

BaseClass = getattr(buffer, 'Buffering', getattr(buffer, 'Buffer', None))

class Feedback_supression(BaseClass):
    """Subclase corregida para inyectar bloques de audio reales al filtro"""
    def __init__(self):
        super().__init__()

        # Inicializamos un bloque vacío con ceros para que record() tenga señal 
        # de referencia desde el segundo cero y el filtro empiece a entrenar de inmediato.
        # Intentamos obtener frames_per_chunk del framework, si no usamos 1024 por defecto.
        chunk_cells = getattr(args, 'frames_per_chunk', 1024)
        self.last_played_chunk = np.zeros(chunk_cells, dtype=np.int16)

        if args.echo_suppression:
            # Inicializamos el filtro adaptativo adaptado al framework
            self.nlms = NLMS(L=chunk_cells, mu=args.nlms_mu)
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
        """Captura el bloque del micrófono y le resta el eco del altavoz"""
        recorded_chunk = super().record()

        # El filtro ahora siempre se ejecutará porque last_played_chunk nunca es None
        if self.nlms is not None:
            # Si el bloque del micrófono viene en estéreo o plano, lo aplanamos para el filtro lineal
            d = recorded_chunk.flatten()
            x = self.last_played_chunk.flatten()

            clean_flat = self.nlms.process(x, d)
            # Devolvemos el audio limpio reconstruyendo la forma original del bloque
            return clean_flat.reshape(recorded_chunk.shape)

        return recorded_chunk

    def play(self, chunk):
        """Almacena inmediatamente el bloque que se va a reproducir"""
        # Guardamos una copia del bloque antes de enviarlo al hardware del altavoz
        self.last_played_chunk = chunk.copy()
        super().play(chunk)

if __name__ == "__main__":
    intercom = Feedback_supression()
    intercom.run()
