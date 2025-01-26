#include <iostream>

using namespace std;

int main()
{
    /**
     * Spatial noise filter order statics mediante MEAN SHIFT
     * @see https://bbbbyang.github.io/2018/03/19/Mean-Shift-Segmentation/
     * @see https://github.com/bbbbyang/Mean-Shift-Segmentation?tab=readme-ov-file
     * @brief Mean shift es muy usado para segmentación.
     *        La primera idea de mean shift es que estima de la densidad de puntos locales basándose en un método paramétrico usando kernels,
     *        no usando histogramas. La densidad de puntos se usa para filtrar los valores a medida que se recorre la imagen. Si tenemos una lista
     *        de números reales y hacemos un histograma, caemos en la pérdida de información de tener que discretizar nuestro dominio en bins rectangulares,
     *        por lo tanto el histograma dependerá fuertemente del ancho de los bins. Pero si usamos funciones de núcleo (kernel) continuos tendremos
     *        una PDF más suave precisa. Para el caso 2D, en vez de usar rectángulos para los bins podemos usar círculos para contar la densidad local de puntos.
     *        La función de kernel determina cómo se agrupan los puntos vecinos puede ser uniforme, gaussiana... Primero computamos la densidad de toda la imagen
     *        mediante nuestra función de kernel y luego calculamos su gradiente, que nos indica la dirección y magnitud del "movimiento" que debe realizar
     *        un punto para moverse a la región (más) densa más cercana.
     *        Cada punto es arrastrado por una fuerza computable que es función de la densidad de sus vecinos. Calculamos los desplazamientos iterativamente,
     *        hasta que se cumpla cierto criterio convergencia.
     *        Cada punto se trata como un punto de R^5 (coords. x, y, color RGB o Lab). Se dice mean clustering consiste en cada punto debe encontrar su "punto de modo";
     *        en mean shift el objetivo al converger es que todos los puntos hayan encontrado su "punto de modo".
     *        El algoritmo establece 2 anchos de banda (algo así como el nº de vecinos), uno para los cálculos de posición y otro para la intensidad.
     *        El shift value es el promedio de de todos los puntos dentro del ancho de banda.
     *        Los puntos que estén dentro del ancho de banda según los dos criterios se mantienen, el resto se filtran. Los puntos se desplazan así a su "punto de modo"
     *        hasta que tras 5 iteraciones no hay movimiento en ninguna de las 5 dimensiones, momento en el cual dicho "punto de modo" se ha encontrado para ese punto.
     *
     */
    {
        //        cv::Mat Img = cv::imread("/home/alejandro/Imágenes/noisyImages/2024-06-12_08-56_2.png");
        //        cv::resize(Img, Img, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST_EXACT);
        //        cv::imshow("original", Img);
        //        cv::cvtColor(Img, Img, CV_RGB2Lab);
        //        cv::Mat msf = Img.clone();
        //        cv::Mat mss = Img.clone();
        //        MeanShiftFilteringAndSegmentation(msf, 8, 16, true);
        //        cv::cvtColor(msf, msf, CV_Lab2RGB);
        //        cv::imshow("Mean Shift filtering", msf);
        //        MeanShiftFilteringAndSegmentation(mss, 8, 16, false);
        //        cv::cvtColor(mss, mss, CV_Lab2RGB);
        //        cv::imshow("Mean Shift segmentation", mss);
        //        cv::waitKey(0);
    }

    return 0;
}
