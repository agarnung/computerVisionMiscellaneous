/**
 * TODO
 * Traducir split bregman a c++ y escribir teoría en variacional
 * @see https://github.com/ycynyu007/L1-norm-total-variation-inpainting-using-Split_Bregman-Iteration
 */

/// Hay dos opciones:
///     - Convertir las imágenes uchar de entrada a float sin escalar
///     - Hacerlo y escalar por 1 / 255 para convertir el rango dinámico a 0-1
/// En el artículo no reescalan. El valor de los parámetros cambia si se reescala o no.
/// Quizá es mejor no normalizar para poder trabajar transparentemente con cualquier entrada,
/// aunque algunas fuentes recomiendan normalizar para mejorar la convergencia

#include <iostream>
#include <chrono>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>

using namespace cv;

cv::Mat addGaussianNoise(const cv::Mat& image, float sigma)
{
    cv::Mat noise(image.size(), image.type());
    cv::randn(noise, 0, sigma);

    return image + noise;
}

cv::Mat makeMultiChannel(cv::Mat& src, int newChannels)
{
    cv::Mat multiChanel(src.size(), CV_MAKETYPE(src.depth(), newChannels));
    std::vector<cv::Mat> channels(newChannels);
    for (int i = 0; i < newChannels; ++i)
        channels[i] = src;
    cv::merge(channels, multiChanel);
    return multiChanel;
}

// Función para desplazar (similar a circshift en MATLAB)
void shift(const cv::Mat& src, cv::Mat& dst, int shift_x, int shift_y)
{
    dst = cv::Mat::zeros(src.size(), src.type());
    int rows = src.rows;
    int cols = src.cols;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int new_i = (i + shift_y + rows) % rows;
            int new_j = (j + shift_x + cols) % cols;
            dst.at<float>(new_i, new_j) = src.at<float>(i, j);
        }
    }
}

// Operadores de derivada hacia adelante y hacia atrás con condiciones de frontera periódicas
Mat Bx(const Mat& input);
Mat By(const Mat& input);
Mat Fx(const Mat& input);
Mat Fy(const Mat& input);

// Implementaciones optimizadas de operadores diferenciales
Mat Bx(const Mat& input) {
    Mat output(input.size(), CV_32F);
    int cols = input.cols;
    for (int i = 0; i < input.rows; ++i) {
        const float* in = input.ptr<float>(i);
        float* out = output.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            int prev = (j == 0) ? cols-1 : j-1;
            out[j] = in[j] - in[prev];
        }
    }
    return output;
}

Mat By(const Mat& input) {
    Mat output(input.size(), CV_32F);
    int rows = input.rows;
    for (int j = 0; j < input.cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            int prev = (i == 0) ? rows-1 : i-1;
            output.at<float>(i, j) = input.at<float>(i, j) - input.at<float>(prev, j);
        }
    }
    return output;
}

Mat Fx(const Mat& input) {
    Mat output(input.size(), CV_32F);
    int cols = input.cols;
    for (int i = 0; i < input.rows; ++i) {
        const float* in = input.ptr<float>(i);
        float* out = output.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            int next = (j == cols-1) ? 0 : j+1;
            out[j] = in[next] - in[j];
        }
    }
    return output;
}

Mat Fy(const Mat& input) {
    Mat output(input.size(), CV_32F);
    int rows = input.rows;
    for (int j = 0; j < input.cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            int next = (i == rows-1) ? 0 : i+1;
            output.at<float>(i, j) = input.at<float>(next, j) - input.at<float>(i, j);
        }
    }
    return output;
}

Mat splitBregmanFFTROF(const Mat& inputImage, float lambda, float theta, int n_iters)
{
    const auto start{std::chrono::steady_clock::now()};

    // Clonar y añadir padding
    Mat f0 = inputImage.clone();
    int padNum = 10;
    copyMakeBorder(f0, f0, padNum, padNum, padNum, padNum, BORDER_REPLICATE);

    int m = f0.rows;
    int n = f0.cols;
    Mat u = f0.clone();

    // Inicialización de variables
    Mat b1 = Mat::zeros(m, n, CV_32F);
    Mat b2 = Mat::zeros(m, n, CV_32F);
    Mat w1 = Mat::zeros(m, n, CV_32F);
    Mat w2 = Mat::zeros(m, n, CV_32F);

    // Precalcular G de manera vectorizada
    auto computeCosineMatrix = [](int size, float scale, bool trueForRows) -> cv::Mat {
        cv::Mat indices(size, 1, CV_32F);
        for (int i = 0; i < size; ++i)
            indices.at<float>(i, 0) = static_cast<float>(i);
        if (!trueForRows)
            indices = indices.t();
        cv::Mat cos_mat;
        cv::multiply(indices, scale, cos_mat);
        cos_mat.forEach<float>([](float& pixel, const int*) { pixel = std::cos(pixel); });

        return cos_mat;
    };
    cv::Mat cos_i = computeCosineMatrix(m, 2 * CV_PI / m, true);
    cv::Mat cos_j = computeCosineMatrix(n, 2 * CV_PI / n, false);
    cv::Mat G = repeat(cos_i, 1, n) + repeat(cos_j, m, 1) - 2;

    // Precalcular el denominador complejo
    Mat denominator_real;
    multiply(2.0f * theta, G, denominator_real);
    denominator_real = 1.0f - denominator_real;
    Mat denominator_imag = Mat(denominator_real.size(), CV_32F, denominator_real.data);
    Mat denominator_complex;
    std::vector<Mat> channels = {denominator_real, denominator_imag};
    merge(channels, denominator_complex);

    // Pre-asignar matrices temporales
    Mat div_w_b(m, n, CV_32F), g(m, n, CV_32F), g_fft, u_fft;

    for (int step = 1; step <= n_iters; ++step)
    {
        // Actualizar u usando FFT
        Mat w1_b1, w2_b2;
        subtract(w1, b1, w1_b1);
        subtract(w2, b2, w2_b2);
        div_w_b = Bx(w1_b1) + By(w2_b2);
        multiply(theta, div_w_b, g);
        subtract(f0, g, g);

        // Procesamiento FFT
        dft(g, g_fft, DFT_COMPLEX_OUTPUT);
        divide(g_fft, denominator_complex, g_fft);
        dft(g_fft, u, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT);

        // Actualizar w con soft-thresholding
        Mat ux = Fx(u), uy = Fy(u);
        Mat c1, c2;
        add(ux, b1, c1);
        add(uy, b2, c2);

        Mat abs_c;
        magnitude(c1, c2, abs_c);
        abs_c += 1e-10; // Evitar división por cero

        Mat thresholded;
        threshold(abs_c - lambda/theta, thresholded, 0, 0, THRESH_TOZERO);

        multiply(thresholded, c1 / abs_c, w1);
        multiply(thresholded, c2 / abs_c, w2);

        // Actualizar variables de Bregman
        subtract(c1, w1, b1);
        subtract(c2, w2, b2);

        // Visualización opcional cada 10 iteraciones
        if (step % step == 0) {
            Mat u_vis;
            u.convertTo(u_vis, CV_8U, 255);
            imshow("splitBregmanFFTROF", u_vis);
            waitKey(1);
        }
    }

    // Eliminar padding y retornar
    Mat result;
    u(Rect(padNum, padNum, inputImage.cols, inputImage.rows)).copyTo(result);

    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};
    std::cout << "Tiempo procesamiento: " << elapsed_seconds.count() << " s" << std::endl;

    return result;
}

cv::Mat processChannel(const cv::Mat& channel, float lambda, float theta, int n_iters) {
    return splitBregmanFFTROF(channel, lambda, theta, n_iters);
}

int main()
{
    /// Parámetros
    std::string inputPath = "/media/alejandro/DATOS/noisyImages/images.jpeg";
    float sigma = 0.5f;
    float lambda = 30.0f / 55.0f;
    float theta = 5.0f / 355.0f;
    int n_iters = 500;
    bool useLabDenoising = false;

    cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_UNCHANGED);
    std::cout << "Input depth: " << inputImage.depth() << ", channels: " << inputImage.channels() << std::endl;

    cv::resize(inputImage, inputImage, cv::Size(800, 600), 0.0, 0.0, cv::INTER_NEAREST_EXACT);

    inputImage.convertTo(inputImage, CV_32F, 1.0 / 255.0, 0.0);
    {
        cv::Mat inputImage_vis;
        inputImage.convertTo(inputImage_vis, CV_8U, 255.0, 0.0);
        cv::imshow("inputImage", inputImage_vis);
    }

    inputImage = addGaussianNoise(inputImage, sigma);
    {
        cv::Mat noisy_vis;
        inputImage.convertTo(noisy_vis, CV_8U, 255.0, 0.0);
        cv::imshow("noisy", noisy_vis);
    }

    cv::Mat denoisedImage;
    if (inputImage.channels() == 3)
    {
        if (useLabDenoising)
        {
            cv::Mat labImage;
            cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);

            std::vector<cv::Mat> labChannels(3);
            cv::split(labImage, labChannels);

            labChannels[0] = processChannel(labChannels[0], lambda, theta, n_iters);

            // std::cout << "labChannels[0].size(): " << labChannels[0].size << ", labChannels[0].type(): " << labChannels[0].type() << std::endl;
            cv::merge(labChannels, labImage);

            cv::cvtColor(labImage, denoisedImage, cv::COLOR_Lab2BGR);
        }
        else
        {
            std::vector<cv::Mat> channels(3);
            cv::split(inputImage, channels);

            for (int i = 0; i < 3; i++)
                channels[i] = processChannel(channels[i], lambda, theta, n_iters);

            cv::merge(channels, denoisedImage);
        }
    }
    else
        denoisedImage = processChannel(inputImage, lambda, theta, n_iters);

    {
        cv::Mat denoisedImage_vis;
        denoisedImage.convertTo(denoisedImage_vis, CV_8U, 255.0, 0.0);
        cv::imshow("splitBregmanFFTROF", denoisedImage_vis);
    }

    cv::waitKey(0);

    return 0;
}
