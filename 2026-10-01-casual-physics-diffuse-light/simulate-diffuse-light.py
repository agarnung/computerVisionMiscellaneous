import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "outputs"
OUTDIR.mkdir(exist_ok=True)

# Parámetros
N = 1024
sigma_phase = 8
wavelengths = {
    "R": 650e-9,
    "G": 550e-9,
    "B": 450e-9
}

gamma = 0.35 # para visualización

def save_image(img, fname, title):
    plt.figure(figsize=(6,6))
    plt.imshow(img**gamma)
    plt.axis("off")
    plt.title(title)
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

# 1) Referencia: SIN pantalla aleatoria (para delta)
reference = np.zeros((N, N, 3))
delta = np.zeros((N, N))
delta[N//2, N//2] = 1.0  # fuente puntual ideal

for i in range(3):
    fft = np.fft.fftshift(np.fft.fft2(delta))
    reference[..., i] = np.abs(fft)**2

reference /= reference.max()
save_image(reference, OUTDIR / "reference.png", "Referencia (sin pantalla)")

# 2) Cargar una imagen real (en RGB)
img_path = SCRIPT_DIR / "baboon.png" 
img = plt.imread(img_path)

# Verificar que la imagen esté en formato RGB (si es a color)
if img.ndim == 3: # Si la imagen es RGB
    img_rgb = img
else:
    raise ValueError("La imagen debe ser RGB")

# Asegurarse de que la imagen esté dentro del rango [0, 1]
img_rgb = np.clip(img_rgb, 0, 1)

# Redimensionar la imagen a tamaño N x N
img_resized = zoom(img_rgb, (N/img_rgb.shape[0], N/img_rgb.shape[1], 1), order=1)

# 3) Crear la referencia con la fase aleatoria (sin efecto de pantalla aún)
reference_with_screen = np.zeros((N, N, 3))

# Generar una fase aleatoria para la "pantalla"
np.random.seed(0)
phase = gaussian_filter(np.random.randn(N, N), sigma=sigma_phase)
phase = 2*np.pi * phase / np.std(phase)

for i, lam in enumerate(wavelengths.values()):
    field = np.exp(1j * phase)
    fft = np.fft.fftshift(np.fft.fft2(field))
    intensity = np.abs(fft)**2

    # ESCALADO CON λ (esto da el color)
    scale = lam / wavelengths["G"]
    resized = zoom(intensity, scale, order=1)

    # recorte o padding al tamaño original
    s = resized.shape[0]
    if s > N:
        d = (s - N)//2
        resized = resized[d:d+N, d:d+N]
    else:
        pad = (N - s)//2
        resized = np.pad(resized, ((pad, N-s-pad), (pad, N-s-pad)))
    reference_with_screen[..., i] = resized

reference_with_screen /= reference_with_screen.max()
save_image(reference_with_screen, OUTDIR / "reference_with_screen.png", "Referencia con Pantalla Aleatoria")

# 4) Varias pantallas aleatorias 
for k, seed in enumerate([1, 5, 20], start=1):
    np.random.seed(seed)
    phase = gaussian_filter(np.random.randn(N, N), sigma=sigma_phase)
    phase = 2*np.pi * phase / np.std(phase)

    rgb = np.zeros((N, N, 3))

    for i, lam in enumerate(wavelengths.values()):

        field = np.exp(1j * phase)
        fft = np.fft.fftshift(np.fft.fft2(field))
        intensity = np.abs(fft)**2

        # ESCALADO CON λ (esto da el color)
        scale = lam / wavelengths["G"]
        resized = zoom(intensity, scale, order=1)

        # recorte o padding al tamaño original
        s = resized.shape[0]
        if s > N:
            d = (s - N)//2
            resized = resized[d:d+N, d:d+N]
        else:
            pad = (N - s)//2
            resized = np.pad(resized, ((pad, N-s-pad), (pad, N-s-pad)))
        rgb[..., i] = resized

    rgb /= rgb.max()
    save_image(rgb, OUTDIR / f"screen_{k:02d}.png", f"Pantalla aleatoria #{k}")

# 5) Aplicar la pantalla aleatoria a la imagen RGB (nuevo)
# Aplicar la distorsión de la pantalla a la imagen RGB
fft_img = np.fft.fftshift(np.fft.fft2(img_resized))

# Expandir la fase aleatoria para que tenga 3 canales (RGB)
field_expanded = np.stack([field] * 3, axis=-1)  # (1024, 1024, 3)

# Multiplicar la imagen en Fourier por la fase expandida
fft_img_distorted = fft_img * field_expanded  # Ahora los tamaños coinciden

# Realizar la transformada inversa para obtener la imagen distorsionada en el espacio real
img_distorted = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_img_distorted)))**2

# Guardar y mostrar la imagen original y la distorsionada
save_image(img_resized, OUTDIR / "original_image.png", "Imagen Original")
save_image(img_distorted, OUTDIR / "distorted_image.png", "Imagen Distorsionada")

print("Imágenes guardadas en:", OUTDIR)
