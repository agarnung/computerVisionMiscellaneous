/**
 * Instalación en Linux de la API de ImageMagick en C++ y pruebas con sus algoritmos de procesamiento de imagen
 * para tratar de mejorar la calidad del OCR para ULMA, entre otros.
 *
 * Tratar de establecer un nexo entre OpenCV y Magick++.
 *
 * @see Instalación:
 *  http://www.fmwconcepts.com/imagemagick/textcleaner/index.php
 *  https://imagemagick.org/script/install-source.php
 *  https://imagemagick.org/Magick++/Install.html
 *  https://stackoverflow.com/questions/54354631/how-can-i-add-magick-to-qt-creator
 *  https://stackoverflow.com/questions/41841553/convert-magickimage-to-cvmat
 **/

#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <ImageMagick-7/Magick++.h>
#include <ImageMagick-7/MagickCore/magick-type.h>

#define DEBUG_MODE_ON
#define WRITE_MODE_ON
#define PREPROCESS_OCR

int main(int argc, char* argv[])
{
    // Initialise ImageMagick library
    Magick::InitializeMagick(*argv);

    // Create Magick++ Image object and read image file
    Magick::Image image("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/antes_.png");
    image.grayscale(Magick::PixelIntensityMethod::UndefinedPixelIntensityMethod);
#ifdef DEBUG_MODE_ON
    std::cout << "image" << std::endl;
    image.display();
#endif

#ifndef PREPROCESS_OCR
    /*
     * Añadir ruido
     */
    //    Magick::Image noise = image;
    //    noise.addNoise(Magick::NoiseType::RandomNoise, 1.0);
    //    std::cout << "RandomNoise" << std::endl;
    //    noise.display();
    //    noise = image;
    //    noise.addNoise(Magick::NoiseType::ImpulseNoise, 1.0);
    //    std::cout << "ImpulseNoise" << std::endl;
    //    noise.display();
    //    noise = image;
    //    noise.addNoise(Magick::NoiseType::PoissonNoise, 1.0);
    //    std::cout << "PoissonNoise" << std::endl;
    //    noise.display();
    //    noise = image;
    //    noise.addNoise(Magick::NoiseType::UniformNoise, 1.0);
    //    std::cout << "UniformNoise" << std::endl;
    //    noise.display();
    //    noise = image;
    //    noise.addNoise(Magick::NoiseType::GaussianNoise, 1.0);
    //    std::cout << "GaussianNoise" << std::endl;
    //    noise.display();
    //    noise = image;
    //    noise.addNoise(Magick::NoiseType::LaplacianNoise, 1.0);
    //    std::cout << "LaplacianNoise" << std::endl;
    //    noise.display();
    //    noise = image;
    //    noise.addNoise(Magick::NoiseType::MultiplicativeGaussianNoise, 1.0);
    //    std::cout << "MultiplicativeGaussianNoise" << std::endl;
    //    noise.display();

    /* Magick::ImageStatistics <- para cada canal de la imagen
     *  - area, canal, depth, entropy, kurtosis, maxima, minima, mean, skewness,
     *    standard deviation, sum, sum subed, sum fourth power, sum squared, variance
     */
    //    Magick::ImageStatistics stats;
    //    stats = image.statistics();

    /*
     * Preprocesamiento
     */
    Magick::Image enhanced = image;
    enhanced.enhance();
#ifdef DEBUG_MODE_ON
    std::cout << "enhanced" << std::endl;
    enhanced.display();
#endif
#ifdef WRITE_MODE_ON
    enhanced.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/enhanced.png");
#endif

    Magick::Image contrasted = image;
    contrasted.contrast(true);
#ifdef DEBUG_MODE_ON
    std::cout << "contrasted" << std::endl;
    contrasted.display();
#endif
#ifdef WRITE_MODE_ON
    contrasted.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/contrasted.png");
#endif

    Magick::Image brightCon = image;
    brightCon.brightnessContrast(1.0, 1.0);
#ifdef DEBUG_MODE_ON
    std::cout << "brightCon" << std::endl;
    brightCon.display();
#endif
#ifdef WRITE_MODE_ON
    brightCon.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/brightCon.png");
#endif

    Magick::Image conStretch = image;
    conStretch.contrastStretch(120, 180);
#ifdef DEBUG_MODE_ON
    std::cout << "conStretch" << std::endl;
    conStretch.display();
#endif
#ifdef WRITE_MODE_ON
    conStretch.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/conStretch.png");
#endif

    Magick::Image maskUnsharpened = image;
    maskUnsharpened.unsharpmask(5.0, 3.0, 50.0, 10.0);
#ifdef DEBUG_MODE_ON
    std::cout << "maskUnsharpened" << std::endl;
    maskUnsharpened.display();
#endif
#ifdef WRITE_MODE_ON
    maskUnsharpened.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/maskUnsharpened.png");
#endif

    Magick::Image adaptiveBlurred = image;
    adaptiveBlurred.adaptiveBlur(3.0, 1.0);
#ifdef DEBUG_MODE_ON
    std::cout << "adaptiveBlurred" << std::endl;
    adaptiveBlurred.display();
#endif
#ifdef WRITE_MODE_ON
    adaptiveBlurred.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/adaptiveBlurred.png");
#endif

    Magick::Image equalized = image;
    equalized.equalize();
#ifdef DEBUG_MODE_ON
    std::cout << "equalized" << std::endl;
    equalized.display();
#endif
#ifdef WRITE_MODE_ON
    equalized.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/equalized.png");
#endif

    Magick::Image gammaCorrected = image;
    gammaCorrected.autoGamma();
#ifdef DEBUG_MODE_ON
    std::cout << "gammaCorrected" << std::endl;
    gammaCorrected.display();
#endif
#ifdef WRITE_MODE_ON
    gammaCorrected.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/gammaCorrected.png");
#endif

    Magick::Image autoLeveled = image;
    autoLeveled.autoLevel();
#ifdef DEBUG_MODE_ON
    std::cout << "autoLeveled" << std::endl;
    autoLeveled.display();
#endif
#ifdef WRITE_MODE_ON
    autoLeveled.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/autoLeveled.png");
#endif

    Magick::Image sigmoidalContrasted = image;
    sigmoidalContrasted.sigmoidalContrast(true, 1.0, 65535.0 / 2.0);
#ifdef DEBUG_MODE_ON
    std::cout << "sigmoidalContrasted" << std::endl;
    autoLeveled.display();
#endif
#ifdef WRITE_MODE_ON
    autoLeveled.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/sigmoidalContrasted.png");
#endif

    Magick::Image sharp = image;
    sharp.sharpen(3.0, 1.0);
#ifdef DEBUG_MODE_ON
    std::cout << "sharp" << std::endl;
    sharp.display();
#endif
#ifdef WRITE_MODE_ON
    sharp.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/sharp.png");
#endif

    Magick::Image adaptiveSharp = image;
    adaptiveSharp.adaptiveSharpen();
#ifdef DEBUG_MODE_ON
    std::cout << "adaptiveSharp" << std::endl;
    adaptiveSharp.display();
#endif
#ifdef WRITE_MODE_ON
    adaptiveSharp.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/adaptiveSharp.png");
#endif

    Magick::Image waveletDenoised = image;
    waveletDenoised.waveletDenoise(5.0, 1.0);
#ifdef DEBUG_MODE_ON
    std::cout << "waveletDenoised" << std::endl;
    waveletDenoised.display();
#endif
#ifdef WRITE_MODE_ON
    waveletDenoised.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/waveletDenoised.png");
#endif

    Magick::Image textAA = image;
    textAA.textAntiAlias(true);
#ifdef DEBUG_MODE_ON
    std::cout << "textAA" << std::endl;
    textAA.display();
#endif
#ifdef WRITE_MODE_ON
    textAA.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/textAA.png");
#endif

    Magick::Image blueShifted = image;
    blueShifted.blueShift(2.5);
#ifdef DEBUG_MODE_ON
    std::cout << "blueShift" << std::endl;
    blueShifted.display();
#endif
#ifdef WRITE_MODE_ON
    blueShifted.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/blueShifted.png");
#endif

    Magick::Image charcoaled = image;
    charcoaled.charcoal(1.0, 3.0);
#ifdef DEBUG_MODE_ON
    std::cout << "charcoaled" << std::endl;
    charcoaled.display();
#endif
#ifdef WRITE_MODE_ON
    charcoaled.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/charcoaled.png");
#endif


    /*
     * Binarizar
     */
    Magick::Image bn = image;
    bn.monochrome(true);
#ifdef DEBUG_MODE_ON
    std::cout << "bn" << std::endl;
    bn.display();
#endif
#ifdef WRITE_MODE_ON
    bn.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/bn.png");
#endif

    /*
     * Threshold
     */
    Magick::Image th1 = image;
    th1.adaptiveThreshold(400, 400, 1.0);
#ifdef DEBUG_MODE_ON
    std::cout << "adaptiveThresholded" << std::endl;
    th1.display();
#endif
#ifdef WRITE_MODE_ON
    th1.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/adaptiveThresholded.png");
#endif

    Magick::Image th2 = image;
    th2.autoThreshold(Magick::AutoThresholdMethod::UndefinedThresholdMethod);
#ifdef DEBUG_MODE_ON
    std::cout << "UndefinedThresholdMethod" << std::endl;
    th2.display();
#endif
#ifdef WRITE_MODE_ON
    th2.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/UndefinedThresholdMethod.png");
#endif

    Magick::Image th3 = image;
    th3.autoThreshold(Magick::AutoThresholdMethod::KapurThresholdMethod);
#ifdef DEBUG_MODE_ON
    std::cout << "KapurThresholdMethod" << std::endl; /// El mejor?
    th3.display();
#endif
#ifdef WRITE_MODE_ON
    th3.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/KapurThresholdMethod.png");
#endif

    Magick::Image th4 = image;
    th4.autoThreshold(Magick::AutoThresholdMethod::OTSUThresholdMethod);
#ifdef DEBUG_MODE_ON
    std::cout << "OTSUThresholdMethod" << std::endl;
    th4.display();
#endif
#ifdef WRITE_MODE_ON
    th4.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/OTSUThresholdMethod.png");
#endif

    Magick::Image th5 = image;
    th5.autoThreshold(Magick::AutoThresholdMethod::TriangleThresholdMethod);
#ifdef DEBUG_MODE_ON
    std::cout << "TriangleThresholdMethod" << std::endl;
    th5.display();
#endif
#ifdef WRITE_MODE_ON
    th5.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/TriangleThresholdMethod.png");
#endif


    /*
     * Magick::Image -> cv::Mat
     */
    // Get dimensions of Magick++ Image
    int w = image.columns();
    int h = image.rows();
    // Make OpenCV Mat of same size with 8-bit and 3 channels
    cv::Mat output(h, w, CV_8UC3);
    // Unpack Magick++ pixels into OpenCV Mat structure
    image.write(0, 0, w, h, "BGR", Magick::CharPixel, output.data);
    // Save opencvImage
    cv::imwrite("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/output.png", output);

    /*
     * Comparar convolución OpenCV e ImageMagick
     */
    /// 0.0079114 seconds (x36 veces más rápido)
    const cv::Mat kernel_opencv = (cv::Mat_<double>(3, 3) <<
                                       0,  1, 0,
                                   1, -4, 1,
                                   0,  1, 0);
    cv::Mat result_opencv;
    output.convertTo(output, CV_64F, 1.0 / 255.0, 0.0);
    auto start = std::chrono::steady_clock::now();
    cv::filter2D(output, result_opencv, CV_64F, kernel_opencv, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "OpenCV filter2D() Laplacian took " << elapsed_seconds.count() << " seconds." << std::endl;

    /// 0.288407 seconds
    const double kernel_opencvimagemagick[3][3] = {
        {0,  1, 0},
        {1, -4, 1},
        {0,  1, 0}
    };
    const double* kernel_ptr = &kernel_opencvimagemagick[0][0];
    start = std::chrono::steady_clock::now();
    image.convolve(9, kernel_ptr);
    end = std::chrono::steady_clock::now();
    elapsed_seconds = end - start;
    std::cout << "Magick++ convolve() Laplacian took " << elapsed_seconds.count() << " seconds." << std::endl;
#endif

    /*
     * Pre-process OCR
     */
#ifdef PREPROCESS_OCR
    Magick::Image ocr = image;
#ifdef DEBUG_MODE_ON
    std::cout << "ocr" << std::endl;
    ocr.display();
#endif
    ocr.sigmoidalContrast(true, 5.0);
#ifdef DEBUG_MODE_ON
    std::cout << "sigmoidalContrast" << std::endl;
    ocr.display();
#endif
    ocr.sharpen();
#ifdef DEBUG_MODE_ON
    std::cout << "sharpen" << std::endl;
    ocr.display();
#endif
    ocr.sigmoidalContrast(true, 5.0);
#ifdef DEBUG_MODE_ON
    std::cout << "sigmoidalContrast" << std::endl;
    ocr.display();
#endif
    ocr.adaptiveBlur(20.0, 1.0);
#ifdef DEBUG_MODE_ON
    std::cout << "adaptiveBlur" << std::endl;
    ocr.display();
#endif
    ocr.sigmoidalContrast(true, 5.0);
#ifdef DEBUG_MODE_ON
    std::cout << "sigmoidalContrast" << std::endl;
    ocr.display();
#endif
    ocr.sharpen();
#ifdef DEBUG_MODE_ON
    std::cout << "sharpen" << std::endl;
    ocr.display();
#endif

    //    ocr.display();
    //    Magick::ImageStatistics stats = ocr.statistics();
    //    double min = stats.channel(Magick::PixelChannel::GrayPixelChannel).minima();
    //    double max = stats.channel(Magick::PixelChannel::GrayPixelChannel).maxima();
    //    std::cout << "min " << min << std::endl;
    //    std::cout << "max " << max << std::endl;
    //    ocr.contrastStretch(1.0, 1.0);
    //    ocr.display();

    ocr.autoLevel();
#ifdef DEBUG_MODE_ON
    std::cout << "autoLevel" << std::endl;
    ocr.display();
#endif
//    ocr.autoThreshold(Magick::AutoThresholdMethod::KapurThresholdMethod);
#ifdef DEBUG_MODE_ON
    std::cout << "KapurThresholdMethod" << std::endl;
    ocr.display();
#endif
    ocr.write("/home/alejandro/Escritorio/proyectos/dsi/pruebasimagemagick/ocr.png");
#endif

    /*
     * OCR ULMA Fourier
     */


    return 0;
}
