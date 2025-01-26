#include <iostream>

using namespace std;

int main()
{
    /**
     * Denoising mediante hard-thresholding y soft-thresholding mediante DTC
     * La idea es eliminar los coeficientes de la DCT pequeños, pues serán los que alberguen menos información, pero
     * se perderán detalles agudos, i.e. se reducirá ruido de alta frecuencia pero inevitablemente también señal
     *  - Hard-thresholding: Establece a cero los coeficientes DCT menores que un umbral, eliminando
     *                       agresivamente el ruido, pero puede perder detalles finos.
     *                       DCT_hard_thresholded(x) = {x ​si ∣x∣ > T, 0 si ∣x∣ ≤ T}​
     *  - Soft-thresholding: Reduce los coeficientes pequeños sin eliminarlos completamente a no ser que sean muy pequeños,
     *                       ofreciendo una transición más suave y preservando más detalles.
     *                       DCT_hard_thresholded(x) = sign(x) ⋅ max(0, ∣x∣ − T)
     *  @see https://github.com/sansuiso/ComputersDontSee/blob/master/src/dct_denoising/main_cds_dct_denoising.cpp
     */
    {
        //        cv::Mat image = cv::imread("/home/alejandro/Imágenes/noisyImages/2024-06-12_08-56_3.png", cv::IMREAD_GRAYSCALE);
        //        cv::Size workingSize(512, 512);
        //        cv::resize(image, image,workingSize, 0.0, 0.0, cv::INTER_LINEAR_EXACT);
        //        image.convertTo(image, CV_32F, 1.0 / 255.0);

        //        // Agregar ruido
        //        float sigmaNoise = 1e-1;
        //        cv::Mat noise(image.size(), image.type());
        //        cv::randn(noise, 0, sigmaNoise);
        //        cv::Mat noisy_image = image + noise;

        //        // Hard-thresholding
        //        cv::Mat dct;
        //        cv::dct(noisy_image, dct);
        //        float hard_threshold = 3.2 * sigmaNoise;
        //        for (int y = 0; y < dct.rows; ++y)
        //        {
        //            float* p_x = dct.ptr<float>(y);
        //            for (int x = 0; x < dct.cols; ++x)
        //            {
        //                float absX = std::fabs(p_x[x]);
        //                p_x[x] = (absX > hard_threshold) ? p_x[x] : 0.0f;
        //            }
        //        }
        //        cv::Mat hardCleanf;
        //        cv::idct(dct, hardCleanf);

        //        // Soft-thresholding
        //        cv::Mat softDCT = dct.clone();
        //        float soft_threshold = 1.5 * sigmaNoise;
        //        for (int y = 0; y < softDCT.rows; ++y)
        //        {
        //            float* p_x = softDCT.ptr<float>(y);
        //            for (int x = 0; x < softDCT.cols; ++x)
        //            {
        //                float absX = std::fabs(p_x[x]);
        //                if (absX > 1e-6)
        //                {
        //                    float shrinkage = std::max(0.0f, 1.0f - soft_threshold / absX);
        //                    p_x[x] *= shrinkage;
        //                }
        //                else
        //                {
        //                    p_x[x] = 0.0f;
        //                }
        //            }
        //        }
        //        cv::Mat softCleanf;
        //        cv::idct(softDCT, softCleanf);

        //        cv::imshow("Original Image", image);
        //        cv::imshow("Noisy Image", noisy_image);
        //        cv::imshow("Hard-Threshold Denoised", hardCleanf);
        //        cv::imshow("Soft-Threshold Denoised", softCleanf);
        //        noisy_image.convertTo(noisy_image, CV_8UC1, 255.0);
        //        image.convertTo(image, CV_8UC1, 255.0);
        //        hardCleanf.convertTo(hardCleanf, CV_8UC1, 255.0);
        //        softCleanf.convertTo(softCleanf, CV_8UC1, 255.0);
        //        std::cout << "Noisy image PSNR:\t\t" << cv::PSNR(noisy_image, image) << std::endl;
        //        std::cout << "Reconstruction (Hard) PSNR:\t" << cv::PSNR(hardCleanf, image) << std::endl;
        //        std::cout << "Reconstruction (Soft) PSNR:\t" << cv::PSNR(softCleanf, image) << std::endl;
        //        cv::waitKey(0);
    }

    return 0;
}
