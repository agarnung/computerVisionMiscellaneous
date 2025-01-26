#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>

/**
 * Normalización de formas.
 *
 * La aplicación de cierta transformación (e.g. afín) a una forma binaria (e.g. contorno) puede ayudar
 * a normalizarla ante cambios de perspectiva de una forma que sus ejes principales se alineen con los coordenados,
 * consiguiendo un efecto similar al de la T-Net en PointNet. Es una forma de normalizar la forma de objetos para
 * ser robusto ante la geometría variante en tareas de identificación, por ejemplo.
 *
 * La idea básica es que tras la normalización, el objeto tendrá una matriz de dispersión unitaria multiplicada por
 * una cte., lo que indica que la forma está lo más compacta posible
 *
 * @see Image Processing Principles and Applications, Tinku Acharya & Ajoy K. Ray p. 200
 */
int main()
{
    const std::string digit_name = "W_1";

    std::string save_path = "/opt/proyectos/agarnung.github.io/assets/blog_images/2025-01-20-shape-normalization/";

    cv::Mat image = cv::imread("/media/alejandro/DATOS/cifrasYLetras/cifrasYLetras/" + digit_name + ".png", cv::IMREAD_GRAYSCALE);
    cv::imwrite(save_path + "input_image_" + digit_name + ".png", image);
    image.setTo(0, image < 127);
    cv::resize(image, image, cv::Size(512, 512), 0.0, 0.0, cv::INTER_LINEAR_EXACT);
    const double blancos_pct = (double)cv::countNonZero(image) / image.total();
    blancos_pct >  0.5 ?
        cv::threshold(image, image, 127, 255, cv::THRESH_BINARY_INV) :
        cv::threshold(image, image, 127, 255, cv::THRESH_BINARY);
    image.convertTo(image, CV_64FC1, 1.0 / 255.0, 0.0);
    cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
    cv::imshow("image", image);
    cv::imwrite(save_path + "input_image_bin_" + digit_name + ".png", image * 255);
    cv::waitKey(0);

    const int totalCols = image.cols;
    const int totalRows = image.rows;

    /// Centroide (x̄, ȳ) de la imagen:
    const double* img_ptr = image.ptr<double>(0);
    double sumX{0.0}, sumY{0.0}, sumIntensity{0.0};
    for (int y = 0; y < totalRows; ++y)
    {
        for (int x = 0; x < totalCols; ++x)
        {
            double intensity = img_ptr[y * totalCols + x];
            sumX += x * intensity;
            sumY += y * intensity;
            sumIntensity += intensity;
        }
    }
    const double centroidX = sumX / sumIntensity;
    const double centroidY = sumY / sumIntensity;
    std::cout << "centroid: (" << (int)centroidX << ", " << (int)centroidY << ")" << std::endl;

    /// Computar la matriz de dispersión de forma M
    double sum_m_11{0.0}, sum_m_22{0.0}, sum_m_12_m_21{0.0};
    for (int y = 0; y < totalRows; ++y)
    {
        for (int x = 0; x < totalCols; ++x)
        {
            double intensity = img_ptr[y * totalCols + x];
            sum_m_11 += x * x * intensity;
            sum_m_22 += y * y * intensity;
            sum_m_12_m_21 += x * y * intensity;
        }
    }
    const double m11 = sum_m_11 / sumIntensity - centroidX * centroidX;
    const double m22 = sum_m_22 / sumIntensity - centroidY * centroidY;
    const double m12 = sum_m_12_m_21 / sumIntensity - centroidX * centroidY;
    const double m21 = m12;
    std::cout << "m_11: " << m11 << std::endl;
    std::cout << "m_22: " << m22 << std::endl;
    std::cout << "m12: " << m12 << std::endl;
    std::cout << "m21: " << m21 << std::endl;
    //        cv::Moments moments = cv::moments(image, true);
    //        double centroidX = moments.m10 / moments.m00;
    //        double centroidY = moments.m01 / moments.m00;
    //        double m11 = moments.mu20 / moments.m00 - centroidX * centroidX;
    //        double m22 = moments.mu02 / moments.m00 - centroidY * centroidY;
    //        double m_12_m_21 = moments.mu11 / moments.m00 - centroidX * centroidY;

    cv::Mat M = (cv::Mat_<double>(2, 2) << m11, m12, m21, m22);
    std::cout << "M:\n" << M << std::endl;

    /// Alinear los ejes de coordenadas con los autovectores de M
    /// (trasladar origen al centroide, rotar ejes a los autovectores
    /// y escalar según el valor de los autovalores)

    // Cálculo manual de eigenvalues y eigenvectors:
    //        double lambda1, lambda2;
    //        const double discriminant = m11 * m11 + 4.0 * m12 * m21 - 2.0 * m11 * m22 + m22 * m22;
    //        if (discriminant >= 0)
    //        {
    //            const double sqrt_discriminant = std::sqrt(discriminant);
    //            lambda1 = (m11 + m22 + sqrt_discriminant) / 2.0;
    //            lambda2 = (m11 + m22 - sqrt_discriminant) / 2.0;
    //            std::cout << "lambda1: " << lambda1 << std::endl;
    //            std::cout << "lambda2: " << lambda2 << std::endl;
    //        }
    //        else
    //        {
    //            double parte_real = (m11 + m22) / 2.0;
    //            double parte_imaginaria = std::sqrt(-discriminant) / 2.0;
    //            std::cout << "lambda1: " << parte_real << " + " << parte_imaginaria << "i" << std::endl;
    //            std::cout << "lambda2: " << parte_real << " - " << parte_imaginaria << "i" << std::endl;
    //            std::cout << "autovalores complejos" << std::endl;
    //            exit(-1);
    //        }
    //        /// @see https://www.soest.hawaii.edu/martel/Courses/GG303/Eigenvectors.pdf
    //        cv::Vec2d E1(
    //            m12 / std::sqrt(m12 * m12 - (lambda1 - m11) * (lambda1 - m11)),
    //            m12 / std::sqrt(m12 * m12 - (lambda1 - m22) * (lambda1 - m22))
    //        );
    //        cv::Vec2d E2(
    //            m12 / std::sqrt(m12 * m12 - (lambda2 - m11) * (lambda2 - m11)),
    //            m12 / std::sqrt(m12 * m12 - (lambda2 - m22) * (lambda2 - m22))
    //        );
    //        std::cout << "E1: " << E1 << std::endl;
    //        std::cout << "E2: " << E2 << std::endl;
    //        E1 /= cv::norm(E1);
    //        E2 /= cv::norm(E2);

    // Cálculo de eigenvalues y eigenvectors con OpenCV
    cv::Mat eigenvalues, eigenvectors;
    cv::eigen(M, eigenvalues, eigenvectors);
    double lambda1 = eigenvalues.at<double>(0, 0);
    double lambda2 = eigenvalues.at<double>(1, 0);
    cv::Vec2d E1(eigenvectors.at<double>(0, 0), eigenvectors.at<double>(1, 0));
    cv::Vec2d E2(eigenvectors.at<double>(0, 1), eigenvectors.at<double>(1, 1));
    std::cout << "Eigenvalue lambda1: " << lambda1 << std::endl;
    std::cout << "Eigenvalue lambda2: " << lambda2 << std::endl;
    std::cout << "Eigenvector E1: " << E1 << std::endl;
    std::cout << "Eigenvector E2: " << E2 << std::endl;

    cv::Mat R(2, 2, CV_64FC1);
    R.at<double>(0, 0) = E1[0];
    R.at<double>(0, 1) = E1[1];
    R.at<double>(1, 0) = E2[0];
    R.at<double>(1, 1) = E2[1];
    std::cout << "Matriz de rotación R:" << R << std::endl;

    /// Calcular k dinámicamente para que el área del objeto transformado sea la misma que la original
    const int whitePixelCountOriginal = cv::countNonZero(image);

    cv::Mat normalized;

    double k = 1.0;
    double tolerance = 0.1; // Tolerancia para el ajuste de k
    double error = std::numeric_limits<double>::max();
    int maxIterations = 50;
    int iteration = 0;
    while (error > tolerance && iteration < maxIterations)
    {
        double minX = std::numeric_limits<double>::max();
        double minY = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::min();
        double maxY = std::numeric_limits<double>::min();
        const double* R_ptr = R.ptr<double>(0);

        for (int y = 0; y < totalRows; ++y)
        {
            for (int x = 0; x < totalCols; ++x)
            {
                double x_centered = x - centroidX;
                double y_centered = y - centroidY;

                double x_rotated = R_ptr[0] * x_centered + R_ptr[1] * y_centered;
                double y_rotated = R_ptr[2] * x_centered + R_ptr[3] * y_centered;

                double x_scaled = (k / std::sqrt(lambda1)) * x_rotated;
                double y_scaled = (k / std::sqrt(lambda2)) * y_rotated;

                if (x_scaled < minX) minX = x_scaled;
                if (x_scaled > maxX) maxX = x_scaled;
                if (y_scaled < minY) minY = y_scaled;
                if (y_scaled > maxY) maxY = y_scaled;
            }
        }

        int newCols = static_cast<int>(std::ceil(maxX - minX + 1));
        int newRows = static_cast<int>(std::ceil(maxY - minY + 1));

        normalized = cv::Mat(newRows, newCols, CV_64FC1, cv::Scalar(0));
        double* normalizedPtr = normalized.ptr<double>(0);
        for (int y = 0; y < totalRows; ++y)
        {
            for (int x = 0; x < totalCols; ++x)
            {
                double intensity = img_ptr[y * totalCols + x];

                double x_centered = x - centroidX;
                double y_centered = y - centroidY;

                double x_rotated = R_ptr[0] * x_centered + R_ptr[1] * y_centered;
                double y_rotated = R_ptr[2] * x_centered + R_ptr[3] * y_centered;

                double x_scaled = (k / std::sqrt(lambda1)) * x_rotated;
                double y_scaled = (k / std::sqrt(lambda2)) * y_rotated;

                int newX = static_cast<int>(std::round(x_scaled - minX));
                int newY = static_cast<int>(std::round(y_scaled - minY));

                if (newX >= 0 && newX < newCols && newY >= 0 && newY < newRows)
                    normalizedPtr[newY * newCols + newX] = intensity;
            }
        }

        const int whitePixelCountTransformed = cv::countNonZero(normalized);

        std::cout << "whitePixelCountTransformed: " << whitePixelCountTransformed << std::endl;
        error = std::abs((double)whitePixelCountOriginal - (double)whitePixelCountTransformed) / (double)whitePixelCountOriginal;
        std::cout << "Iteration: " << iteration << ", Error: " << error << std::endl;

        k *= whitePixelCountTransformed < whitePixelCountOriginal ? 1.1 : 0.9;

        ++iteration;
    }

    /// Para refinar el resultado, recortar la bounding box del objeto
    int filaInicio = 0;
    int filaFin = normalized.rows - 1;
    int colInicio = 0;
    int colFin = normalized.cols - 1;
    for (int i = 0; i < normalized.rows; ++i)
    {
        if (cv::countNonZero(normalized.row(i)) > 0)
        {
            filaInicio = i;
            break;
        }
    }
    for (int i = normalized.rows - 1; i >= filaInicio; --i)
    {
        if (cv::countNonZero(normalized.row(i)) > 0)
        {
            filaFin = i;
            break;
        }
    }
    for (int i = 0; i < normalized.cols; ++i)
    {
        if (cv::countNonZero(normalized.col(i)) > 0)
        {
            colInicio = i;
            break;
        }
    }
    for (int i = normalized.cols - 1; i >= colInicio; --i)
    {
        if (cv::countNonZero(normalized.col(i)) > 0)
        {
            colFin = i;
            break;
        }
    }
    cv::Rect roi(colInicio, filaInicio, colFin - colInicio + 1, filaFin - filaInicio + 1);
    cv::imwrite(save_path + "normalized_not_cropped_" + digit_name + ".png", normalized * 255);
    normalized = normalized(roi).clone();
    cv::imwrite(save_path + "normalized_cropped_" + digit_name + ".png", normalized * 255);

    cv::namedWindow("normalized", cv::WINDOW_AUTOSIZE);
    cv::imshow("normalized", normalized);
    cv::waitKey(0);
    cv::imwrite(save_path + "shapenormalized_" + digit_name + ".png", normalized * 255);

    return 0;
}
