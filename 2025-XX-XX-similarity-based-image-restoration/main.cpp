#include <iostream>

using namespace std;

int main()
{
    /**
     * A. Kheradmand and P. Milanfar. A general framework for regularized, similarity-based image restoration. IEEE Transactions on Image Processing, 23(12):5136–5151, 2014.
     * @brief El paper trata de la restauración de imágenes. Su idea es ver la imagen como un grafo ponderado; la estructura de la imagen se recupera basándose en similitud de
     *        kernel y la graph Laplacian normalizada, la cual evoca un nuevo data term y regularization term. Los coeficientes usados para describir la matriz se hacen de
     *        manera beneficiosa y por tanto la matriz es simétrica, definida positiva y otros atributos buenos para la convexidad/minimización del funcional.
     *        Se resuelve la optimización mediante gradientes conjugados y en cada iteración se recomputan los pesos de la matriz Laplaciana.
     *        La forma del funcional propuesto permite además analizar el espectro de la PDE a través de los valores propios de la Laplaciana.
     *        El método permite distintos tipos de restauraciones, e.g. deblurring, denoising y sharpening.
     */
    {

    }

    return 0;
}
