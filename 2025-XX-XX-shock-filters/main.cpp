#include <iostream>

using namespace std;

int main()
{

    /**
     * Shock Filters (sharpening == edge enhancement)
     *
     * Este filtro aplica una evolución basada en PDE hiperbólica (como las de onda) para sharpening, i.e. potenciación de los bordes
     *
     *      df/dt = -sign(delta(f)) * norm(grad(f)),
     *
     * donde:
     * - grad(f) = [fx, fy]^T representa el gradiente de la imagen.
     * - norm(grad(f)) = sqrt(fx^2 + fy^2) es la magnitud del gradiente.
     * - delta(f) es una aproximación del Laplaciano basada en derivadas cruzadas.
     *
     * El filtro de choque se aplica iterativamente, aumentando la nitidez en las zonas de borde y suavizando las regiones planas.
     *
     * @see S. Osher and L.I. Rudin. Feature-oriented image enhancement using shock ﬁlters. SIAM Journal of Numerical Analysis, 27(4):919–940, August 1990.
     * @see Mathematical Image Processing pág. 141
     * @see Jia, "Two-Phase Kernel Estimation for Robust Motion Deblurring", 2013.
     * @see https://github.com/apvijay/shock_filter/blob/master/shock_filter.m
     */
    {
        //        cv::Mat f = cv::imread("/home/alejandro/Imágenes/blurry_images/blurryBuilding.png", cv::IMREAD_GRAYSCALE);
        //        f.convertTo(f, CV_32FC1, 1.0 / 255.0, 0.0);
        //        cv::resize(f , f, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST_EXACT);
        ////        cv::GaussianBlur(f, f, cv::Size(9, 9), 0);
        //        cv::Mat sharpened = shockFilter(f, 30, 0.1);
    }
    return 0;
}
