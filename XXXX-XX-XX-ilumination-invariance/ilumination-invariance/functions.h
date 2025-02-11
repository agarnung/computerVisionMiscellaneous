#pragma once

#include <common/cv/matrixfunctions.h>
#include <common/cv/fourier.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xphoto.hpp>

#include <eigen3/Eigen/Dense>

cv::Mat hamming_window(const cv::Size& size)
{
    cv::Mat hammingX = cv::Mat::zeros(1, size.width, CV_32FC1);
    cv::Mat hammingY = cv::Mat::zeros(size.height, 1, CV_32FC1);

    for (int i = 0; i < size.width; ++i)
        hammingX.at<float>(0, i) = 0.54 - 0.46 * std::cos(2 * CV_PI * i / (size.width - 1));

    for (int i = 0; i < size.height; ++i)
        hammingY.at<float>(i, 0) = 0.54 - 0.46 * std::cos(2 * CV_PI * i / (size.height - 1));

    return hammingY * hammingX;
}

cv::Mat fourier_highpass_invariant(const cv::Mat& image, int cutoff = 30)
{
    cv::Mat imgFloat;
    image.convertTo(imgFloat, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(imgFloat, channels);

    int h = image.rows;
    int w = image.cols;

    cv::Mat highpass_img(image.size(), CV_32FC3);

    for (int i = 0; i < 3; i++)
    {
        cv::Mat f_transform;
        cv::dft(channels[i], f_transform, cv::DFT_COMPLEX_OUTPUT);
        cv::Mat f_transform_shifted = dsi::Fourier::fftshift(f_transform);

        cv::Mat planes[2];
        cv::split(f_transform_shifted, planes);

        cv::Mat high_pass_filter = cv::Mat::zeros(h, w, CV_32FC1);
        for (int u = 0; u < h; ++u)
        {
            for (int v = 0; v < w; ++v)
            {
                float radius = std::sqrt(std::pow(u - h / 2, 2) + std::pow(v - w / 2, 2));
                if (radius > cutoff)
                    high_pass_filter.at<float>(u, v) = 1.0;
            }
        }

        cv::Mat window = hamming_window(image.size());
        high_pass_filter = high_pass_filter.mul(window);

        planes[0] = planes[0].mul(high_pass_filter);
        planes[1] = planes[1].mul(high_pass_filter);

        cv::Mat filtered_shifted;
        cv::merge(planes, 2, filtered_shifted);
        cv::Mat filtered = dsi::Fourier::fftshift(filtered_shifted);

        cv::Mat f_inverse;
        cv::dft(filtered, f_inverse, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

        cv::normalize(f_inverse, f_inverse, 0, 1, cv::NORM_MINMAX);

        channels[i] = f_inverse.clone();
    }

    cv::merge(channels, highpass_img);

    return highpass_img;
}

cv::Mat pca_ii(const cv::Mat& image)
{
    cv::Mat imgFloat;
    image.convertTo(imgFloat, CV_32FC3, 1.0 / 255.0);

    if (imgFloat.channels() != 3)
        throw std::invalid_argument("La imagen debe ser RGB con tres canales.");

    int h = imgFloat.rows;
    int w = imgFloat.cols;

    cv::Mat pixels = imgFloat.reshape(1, h * w);

    cv::Mat geoMean;
    cv::pow(pixels, 1.0 / 3, geoMean);

    cv::Mat ch_r = pixels.col(0) / (geoMean.col(0) + 1e-10);
    cv::Mat ch_b = pixels.col(2) / (geoMean.col(2) + 1e-10);

    cv::log(ch_r, ch_r);
    cv::log(ch_b, ch_b);

    Eigen::MatrixXd X(ch_r.rows, 2);
    for (int i = 0; i < ch_r.rows; ++i)
    {
        X(i, 0) = ch_r.at<float>(i);
        X(i, 1) = ch_b.at<float>(i);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd vec = svd.matrixV().col(0);

    // Ángulo alpha del primer componente principal
    double alpha = atan2(-vec(0), vec(1));

    cv::Mat ii_image = ch_r * cos(alpha) + ch_b * sin(alpha);
    cv::Mat ii_resized = ii_image.reshape(1, h);

    double minVal, maxVal;
    cv::minMaxLoc(ii_resized, &minVal, &maxVal);
    ii_resized = (ii_resized - minVal) / (maxVal - minVal);

    return ii_resized;
}

float log_app(float x)
{
    const float alpha = 5000.0f;
    const float alpha_inv = 1.0f / alpha;
    return alpha * (std::pow(x, alpha_inv) - 1);
}

cv::Mat GetInvariantImage(const cv::Mat& input_image, float angle, int tipus = 1)
{
    cv::Mat imatge;
    input_image.convertTo(imatge, CV_32F, 1.0 / 255.0);
    imatge.setTo(cv::Scalar(1e-10), imatge == 0);

    cv::Mat invariant = cv::Mat::zeros(imatge.size(), CV_32F);

    const cv::Vec3f* imatge_ptr = (const cv::Vec3f*)imatge.data;
    float* invariant_ptr = (float*)invariant.data;

    for (int i = 0; i < imatge.rows; ++i)
    {
        for (int j = 0; j < imatge.cols; ++j)
        {
            int index = i * imatge.cols + j;
            cv::Vec3f pixel = imatge_ptr[index];
            float R = pixel[0];
            float G = pixel[1];
            float B = pixel[2];
            float GeomMean = std::pow(R * G * B, 1.0 / 3.0); // Media geométrica
            if (tipus == 0)
                invariant_ptr[index] = std::cos(angle * CV_PI / 180.0) * log_app(R / G) +std::sin(angle * CV_PI / 180.0) * log_app(B / G);
            else if (tipus == 1)
                invariant_ptr[index] = std::cos(angle * CV_PI / 180.0) * log_app(R / GeomMean) + std::sin(angle * CV_PI / 180.0) * log_app(B / GeomMean);
            else if (tipus == 2)
                invariant_ptr[index] = std::cos(angle * CV_PI / 180.0) * log_app(R / G) + std::sin(angle * CV_PI / 180.0) * log_app(B / G);
        }
    }

    return invariant;
}

// Convert RGB image to illumination invariant image using Maddern et al. (2014).
cv::Mat maddern2014(const cv::Mat& image, float alpha)
{
    cv::Mat image_float;
    image.convertTo(image_float, CV_32FC1, 1.0 / 255.0);

    const float epsilon = 1e-10;
    cv::Mat ii_image = cv::Mat::zeros(image.size(), CV_32FC1);
    float* image_ptr = (float*)image_float.data;
    float* ii_image_ptr = (float*)ii_image.data;
    for (int y = 0; y < ii_image.rows; ++y) {
        for (int x = 0; x < ii_image.cols; ++x) {
            int index = y * ii_image.cols * 3 + x * 3;
            float B = image_ptr[index];
            float G = image_ptr[index + 1];
            float R = image_ptr[index + 2];
            ii_image_ptr[y * ii_image.cols + x] = 0.5 + std::log(G + epsilon) -
                                                  alpha * std::log(B + epsilon) -
                                                  (1 - alpha) * std::log(R + epsilon);
        }
    }

    //    double minVal, maxVal;
    //    cv::minMaxLoc(ii_image, &minVal, &maxVal);
    //    ii_image = (ii_image - minVal) / (maxVal - minVal);
    //    cv::normalize(ii_image, ii_image, 0, 1, cv::NORM_MINMAX);

    return ii_image;
}

// Implement the algorithm proposed by Alvarez and Lopez (2011).
cv::Mat alvarez2011(const cv::Mat& image, float alpha, bool inv = false)
{
    cv::Mat ii_image = GetInvariantImage(image, alpha * 360, 0);

    if (inv)
        ii_image = 1 - ii_image;

    //    double minVal, maxVal;
    //    cv::minMaxLoc(ii_image, &minVal, &maxVal);
    //    ii_image = (ii_image - minVal) / (maxVal - minVal);
    //    cv::normalize(ii_image, ii_image, 0, 1, cv::NORM_MINMAX);

    return ii_image;
}

// Implement the algorithm proposed by Zhenqiang Ying et al. (2015).
cv::Mat ying2015(const cv::Mat& image, bool inv = false)
{
    cv::Mat floatImage;
    image.convertTo(floatImage, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(floatImage, channels);

    cv::Mat B = channels[2];

    cv::Mat RGB_max = cv::Mat::zeros(B.size(), CV_32F);
    float* R_ptr = (float*)channels[0].data;
    float* G_ptr = (float*)channels[1].data;
    float* B_ptr = (float*)channels[2].data;
    float* RGB_max_ptr = (float*)RGB_max.data;
    for (int y = 0; y < floatImage.rows; ++y)
    {
        for (int x = 0; x < floatImage.cols; ++x)
        {
            int index = y * floatImage.cols + x;
            float R = R_ptr[index];
            float G = G_ptr[index];
            float B_val = B_ptr[index];
            float max_val = std::max(R, std::max(G, B_val));
            RGB_max_ptr[index] = max_val;
        }
    }
    cv::Mat ii_image = (RGB_max - B) / (RGB_max + std::numeric_limits<float>::epsilon()); // Característica S'

    if (inv)
        ii_image = 1 - ii_image;

    //     double minVal, maxVal;
    //     cv::minMaxLoc(ii_image, &minVal, &maxVal);
    //     ii_image = (ii_image - minVal) / (maxVal - minVal);

    return ii_image;
}

// Implement the algorithm proposed by Zhenqiang Ying et al. (2016).
cv::Mat ying2016(const cv::Mat& image, float bias, bool inv = false)
{
    cv::Mat normalized_image;
    image.convertTo(normalized_image, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(normalized_image, channels);

    cv::Mat B = channels[0];
    cv::Mat G = channels[1];

    cv::Mat ii_image = 2 - (G + bias) / (B + std::numeric_limits<float>::epsilon());

    cv::minMaxLoc(ii_image, nullptr, nullptr, nullptr, nullptr, cv::Mat()); // Solo para asegurar el rango
    ii_image = cv::min(cv::max(ii_image, 0), 1);  // Limitar valores entre 0 y 1

    if (inv)
        ii_image = 1 - ii_image;

    return ii_image;
}
