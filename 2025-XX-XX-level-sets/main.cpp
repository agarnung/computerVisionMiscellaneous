#include <iostream>

using namespace std;

int main()
{
    /**
     * Mostrar una al lado de otra la imagen gris original y sus isofotas (level sets) a la derecha, o al menos algunos de ellos (0-255).
     * Y hacer pequeña prueba para demostrar que en imágenes binarias la minimización de la TV es equivalente a la minimización de la
     * longitud del (único) contorno binario (fórmula de co-área).
     */
    {
        cv::Mat img = cv::imread("/home/alejandro/Imágenes/ImageProcessingSamples/peppers3.tif", cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(800, 800), 0.0, 0.0, cv::INTER_NEAREST_EXACT);
        int num_levels = 15;
        showLevelSets(img, num_levels);
    }

    return 0;
}
