import numpy as np
from scipy.io.wavfile import write, read
from scipy.signal import medfilt

frecuencias = [261.63, 329.62, 392.00]
duracion = 2.0
sample_rate = 44100

t = np.linspace(0, duracion, int(sample_rate * duracion), endpoint=False)
señal = sum(np.sin(2 * np.pi * f * t) for f in frecuencias)
señal /= np.max(np.abs(señal))

fade_time = 0.05
fade_samples = int(sample_rate * fade_time)
fade_in = np.linspace(0, 1, fade_samples)
fade_out = np.linspace(1, 0, fade_samples)
señal[:fade_samples] *= fade_in
señal[-fade_samples:] *= fade_out

señal_int16 = np.int16(señal * 32767)
write('acorde_do_mayor.wav', sample_rate, señal_int16)

# --- Generar señal corrompida ---

ruido_gaussiano = np.random.normal(0, 0.02, size=señal.shape)
señal_ruido = señal + ruido_gaussiano

num_impulsos = int(0.005 * len(señal))
indices_impulsos = np.random.choice(len(señal), num_impulsos, replace=False)
señal_ruido[indices_impulsos] = np.random.choice([-1, 1], size=num_impulsos)

señal_ruido /= np.max(np.abs(señal_ruido))
señal_ruido_int16 = np.int16(señal_ruido * 32767)
write('acorde_do_mayor_ruido.wav', sample_rate, señal_ruido_int16)

# --- Aplicar filtro de mediana ---

_, señal_ruido_leida = read('acorde_do_mayor_ruido.wav')

# Trabajar en float (-1.0, 1.0)
señal_ruido_leida = señal_ruido_leida.astype(np.float32) / 32767

# Aplicar filtro de mediana
kernel_size = 5
señal_filtrada = medfilt(señal_ruido_leida, kernel_size=kernel_size)

# Normalizar de nuevo
señal_filtrada /= np.max(np.abs(señal_filtrada))

# Guardar señal filtrada
señal_filtrada_int16 = np.int16(señal_filtrada * 32767)
write('acorde_do_mayor_filtrado.wav', sample_rate, señal_filtrada_int16)

print("Archivos 'acorde_do_mayor.wav', 'acorde_do_mayor_ruido.wav' y 'acorde_do_mayor_filtrado.wav' creados.")

