/**
 * Normalizar la iluminación de imágenes a color (en principio) basándose en
 * la eficiencia cuántica de los sensores de imagen típicos y otros métodos
 * @todo https://github.com/hangong/gfinvim
 */

#include <functions.h>

int main()
{
    cv::Mat img = cv::imread("/home/alejandro/Imágenes/CIE Norma fotos/23_2023-11-17_23-28-23_StepAcq_2321232628_2023-11-17_23-28-23_scans_scan2_20231117232825_0.png", CV_LOAD_IMAGE_UNCHANGED);

    const int size = 512;
    cv::resize(img, img, cv::Size(size, size), 0.0, 0.0, cv::INTER_NEAREST_EXACT);

    cv::imshow("img", img);
    cv::waitKey(0);

    float alpha = 0.5f; // Camera alpha para maddern y alvarez
    float bias = 0.5f;  // Bias para Ying2016

    cv::Mat alvarez_image = alvarez2011(img, alpha, false);
    cv::Mat maddern_image = maddern2014(img, alpha);
    cv::Mat ying2015_image = ying2015(img, false);
    cv::Mat ying2016_image = ying2016(img, bias, false);
    cv::Mat pca_image = pca_ii(img);
    cv::Mat highpass_image = fourier_highpass_invariant(img, 20.0f);

    cv::imshow("Alvarez 2011 Image", alvarez_image);
    cv::imshow("Maddern 2014 Image", maddern_image);
    cv::imshow("Ying 2015 Image", ying2015_image);
    cv::imshow("Ying 2016 Image", ying2016_image);
    cv::imshow("PCA Image", pca_image);
    cv::imshow("Fourier Highpass Image", highpass_image);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
