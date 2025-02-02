#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/ximgproc.hpp>
#include <chrono>

cv::Mat addGaussianNoise(const cv::Mat& image, double mean = 0.0, double stddev = 0.1)
{
    cv::Mat noisyImage = image.clone();
    cv::Mat noise(image.size(), image.type());

    cv::randn(noise, mean, stddev);

    noisyImage += noise;

    return noisyImage;
}

int main()
{
    std::string imagePath = "/home/alejandro/Pictures/bike.jpg";
    double gaussianMean = 0.0, gaussianStddev = 0.25;
    double sigmaS = 16.0, sigmaC = 0.2;
    int bilateralKernelSize = 55;

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    image.convertTo(image, CV_32FC1, 1.0 / 255.0);

    image = addGaussianNoise(image, gaussianMean, gaussianStddev);

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat manifoldFiltered;
    // cv::Ptr<cv::ximgproc::AdaptiveManifoldFilter> manifoldFilter = cv::ximgproc::AdaptiveManifoldFilter::create();
    // manifoldFilter->setSigmaR(sigmaC);
    // manifoldFilter->setSigmaS(sigmaS);
    // manifoldFilter->filter(image, manifoldFiltered);
    cv::ximgproc::amFilter(image, image, manifoldFiltered, sigmaS, sigmaC, true);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double amTime = duration.count() / 1000.0;
    std::cout << "Time taken by Adaptive Manifold Filter: " << duration.count() << " ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();

    cv::Mat bilateralFiltered;
    cv::bilateralFilter(image, bilateralFiltered, bilateralKernelSize, sigmaC, sigmaS);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    double bilateralTime = duration.count() / 1000.0;
    std::cout << "Time taken by Bilateral Filter: " << duration.count() << " ms" << std::endl;

    double m_psnr = cv::PSNR(image, manifoldFiltered, 1.0);
    double b_psnr = cv::PSNR(image, bilateralFiltered, 1.0);

    std::cout << "Manifold PSNR: " << m_psnr << " dB" << std::endl;
    std::cout << "Bilateral PSNR: " << b_psnr << " dB" << std::endl;
    cv::imshow("Noisy Image", image);
    cv::imshow("Manifold Filtered Image, PSNR: " + std::to_string(m_psnr) + ", Elapsed time: " + std::to_string(amTime) + "s", manifoldFiltered);
    cv::imshow("Bilateral Filtered Image, PSNR: " + std::to_string(b_psnr) + ", Elapsed time: " + std::to_string(bilateralTime) + "s", bilateralFiltered);
    cv::waitKey(0);

    return 0;
}
