import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def enhance_contrast_with_plane(img_gray: np.ndarray,
                                alpha: float = 1.0,
                                tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """
    Realza el contraste de una imagen en escala de grises ajustándole un plano.

    Parámetros:
    -----------
    img_gray : np.ndarray
        Imagen de entrada en escala de grises (dtype uint8 o float).
    alpha : float
        Factor de escala para la contribución del plano (por defecto 1.0).
    tol : float
        Tolerancia para evitar división por cero al centrar (por defecto 1e-8).

    Retorna:
    --------
    img_out : np.ndarray
        Imagen de salida (uint8) con contraste mejorado.
    plane : np.ndarray
        Plano ajustado en la imagen original (float32, sin escalar).
    """
    # Convertir a float32
    I = img_gray.astype(np.float32)
    h, w = I.shape

    # Generar coordenadas X, Y
    ys, xs = np.mgrid[0:h, 0:w]

    # Preparar sistema para regresión lineal: A * [a, b, c] = I.ravel()
    A = np.column_stack((xs.ravel(), ys.ravel(), np.ones(h*w, dtype=np.float32)))
    b = I.ravel()

    # Resolver mínimos cuadrados
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = coeffs

    # Reconstruir el plano ajustado
    plane = (a * xs + b_ * ys + c).astype(np.float32)

    # Centrar el plano para variación ±1
    plane_centered = plane - np.mean(plane)
    scale = max(np.std(plane_centered), tol)
    plane_centered /= scale

    # Aplicar plano al contraste original
    I_out = I - alpha * plane_centered * 127
    I_out = np.clip(I_out, 0, 255).astype(np.uint8)

    return I_out, plane

def show_3d_surface(img: np.ndarray, plane: np.ndarray, title: str = "Original image heightmap and fitted plane"):
    h, w = img.shape
    ys, xs = np.mgrid[0:h, 0:w]
    step_y = max(h // 100, 1)
    step_x = max(w // 100, 1)
    xs3d = xs[::step_y, ::step_x]
    ys3d = ys[::step_y, ::step_x]
    img3d = img.astype(np.float32)[::step_y, ::step_x]
    plane3d = plane[::step_y, ::step_x]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xs3d, ys3d, img3d, rstride=1, cstride=1, cmap='gray', alpha=0.8)
    ax.plot_surface(xs3d, ys3d, plane3d, rstride=1, cstride=1, color='red', alpha=1.0)
    ax.set_title(title)
    ax.set_xlabel('X (columns)')
    ax.set_ylabel('Y (rows)')
    ax.set_zlabel('Intensity')
    plt.tight_layout()
    plt.savefig(f"3d_surface_{number_file}.png", dpi=400)
    plt.show()

if __name__ == "__main__":
    img_path = "/opt/agarnung.github.io/assets/blog_images/2025-04-22-naive-even-ilumination/8.png"
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    number_file = os.path.splitext(os.path.basename(img_path))[0]

    if img is None:
        raise ValueError("No se pudo cargar la imagen.")

    is_gray = len(img.shape) == 2 or img.shape[2] == 1

    alpha=0.25

    if is_gray:
        print("Imagen en escala de grises detectada.")
        gray = img if len(img.shape) == 2 else img[:, :, 0]
        enhanced, plane = enhance_contrast_with_plane(gray, alpha=alpha)
        output_image = enhanced
        gray_for_3d = gray  # para visualización
    else:
        print("Imagen en color detectada.")
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(img_lab)
        L_enhanced, plane = enhance_contrast_with_plane(L, alpha=alpha)
        img_lab_enhanced = cv2.merge([L_enhanced, a, b])
        output_image = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_Lab2BGR)
        gray_for_3d = L  # usar canal L para visualización

    # Guardar imagen mejorada
    output_filename = f"perceptually_improved_{number_file}.png"
    cv2.imwrite(output_filename, output_image)

    show_3d_surface(gray_for_3d, plane)

    # Mostrar antes y después
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    if is_gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    if is_gray:
        plt.imshow(output_image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Image with even(ed) illumination")
    plt.axis('off')
    plt.suptitle("Before and after", fontsize=14)
    plt.tight_layout()
    plt.show()