#include <iostream>
#include <random>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>

// Función que genera y muestra los conjuntos de nivel con colores distintos
cv::Mat showLevelSets(const cv::Mat& img, double& totalLength, int numLevels, int thickness = 1) {
    if (img.empty() || img.channels() != 1) {
        std::cerr << "La imagen debe ser en escala de grises." << std::endl;
        return cv::Mat();
    }

    if (numLevels < 3) {
        std::cerr << "Mínimo 3 level sets" << std::endl;
        return cv::Mat();
    }

    /// Definir criterio de comparación para el std::set, que no permite duplicados. cv::Scalar no tiene
    /// criterio de comparación por defecto
    struct ScalarComparator {
        bool operator()(const cv::Scalar& a, const cv::Scalar& b) const {
            if (a[0] != b[0]) return a[0] < b[0];
            if (a[1] != b[1]) return a[1] < b[1];
            return a[2] < b[2];
        }
    };

    totalLength = 0.0;

    cv::Mat displayImage = img.clone();
    cv::cvtColor(img, displayImage, cv::COLOR_GRAY2BGR);

    std::set<cv::Scalar, ScalarComparator> usedColors;

    auto getUniqueColor = [&usedColors]() -> cv::Scalar {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        cv::Scalar color;
        do {
            color = cv::Scalar(dis(gen), dis(gen), dis(gen));
        } while (usedColors.find(color) != usedColors.end());

        usedColors.insert(color);
        return color;
    };

    for (int i = 0; i < numLevels; ++i)
    {
        int threshold_value = (i * 255) / (numLevels - 1);

        cv::Scalar color = getUniqueColor();
        std::cout << "Level " << i << ": Threshold value = " << threshold_value
                  << " | Color = (" << color[0] << ", " << color[1] << ", " << color[2] << ")" << std::endl;

        cv::Mat binaryImage;
        cv::threshold(img, binaryImage, threshold_value, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);

        for (const auto& contour : contours) {
            totalLength += cv::arcLength(contour, true);
        }

        for (size_t j = 0; j < contours.size(); ++j) {
            cv::drawContours(displayImage, contours, (int)j, color, thickness, cv::LINE_AA);
        }
    }

    cv::imshow("Original with Level Sets", displayImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return displayImage;
}

cv::Mat showLevelSets(const cv::Mat& img, double& totalLength, const std::vector<int>& intensityLevels, int thickness = 1) {
    if (img.empty() || img.channels() != 1) {
        std::cerr << "La imagen debe ser en escala de grises." << std::endl;
        return cv::Mat();
    }

    if (intensityLevels.size() < 2) {
        std::cerr << "Debe haber al menos 2 level sets." << std::endl;
        return cv::Mat();
    }

    totalLength = 0.0;

    cv::Mat displayImage;
    cv::cvtColor(img, displayImage, cv::COLOR_GRAY2BGR);

    /// Definir criterio de comparación para el std::set, que no permite duplicados. cv::Scalar no tiene
    /// criterio de comparación por defecto
    struct ScalarComparator {
        bool operator()(const cv::Scalar& a, const cv::Scalar& b) const {
            if (a[0] != b[0]) return a[0] < b[0];
            if (a[1] != b[1]) return a[1] < b[1];
            return a[2] < b[2];
        }
    };

    std::set<cv::Scalar, ScalarComparator> usedColors;

    auto getUniqueColor = [&usedColors]() -> cv::Scalar {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);

        cv::Scalar color;
        do {
            color = cv::Scalar(dis(gen), dis(gen), dis(gen));
        } while (usedColors.find(color) != usedColors.end());

        usedColors.insert(color);
        return color;
    };

    for (size_t i = 0; i < intensityLevels.size(); ++i) {
        int threshold_value = intensityLevels[i];

        cv::Scalar color = getUniqueColor();
        std::cout << "Level " << i << ": Threshold value = " << threshold_value
                  << " | Color = (" << color[0] << ", " << color[1] << ", " << color[2] << ")" << std::endl;

        cv::Mat binaryImage;
        cv::threshold(img, binaryImage, threshold_value, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        for (const auto& contour : contours) {
            totalLength += cv::arcLength(contour, true);
        }

        for (size_t j = 0; j < contours.size(); ++j) {
            cv::drawContours(displayImage, contours, (int)j, color, thickness, cv::LINE_AA);
        }
    }

    cv::imshow("Original with Level Sets", displayImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return displayImage;
}

double computeTotalVariationIsotropic(const cv::Mat& img)
{
    if (img.empty() || img.channels() != 1) {
        std::cerr << "La imagen debe estar en escala de grises." << std::endl;
        return 0.0;
    }

    double totalVariation = 0.0;
    int rows = img.rows;
    int cols = img.cols;

    for (int y = 0; y < rows - 1; ++y) {  // sin borde inferior
        for (int x = 0; x < cols - 1; ++x) {  // sin borde derecho
            // Diferencias hacia adelante
            double dx = img.at<uchar>(y, x + 1) - img.at<uchar>(y, x);  // Derivada en x
            double dy = img.at<uchar>(y + 1, x) - img.at<uchar>(y, x);  // Derivada en y

            totalVariation += std::sqrt(dx * dx + dy * dy);  // Suma las diferencias
        }
    }

    return totalVariation;
}

double computeTotalVariationAnisotropic(const cv::Mat& img)
{
    if (img.empty() || img.channels() != 1) {
        std::cerr << "La imagen debe estar en escala de grises." << std::endl;
        return 0.0;
    }

    double totalVariation = 0.0;
    int rows = img.rows;
    int cols = img.cols;

    for (int y = 0; y < rows - 1; ++y) {  // sin borde inferior
        for (int x = 0; x < cols - 1; ++x) {  // sin borde derecho
            // Diferencias hacia adelante
            double dx = std::abs(img.at<uchar>(y, x + 1) - img.at<uchar>(y, x));  // Derivada en x (valor absoluto)
            double dy = std::abs(img.at<uchar>(y + 1, x) - img.at<uchar>(y, x));  // Derivada en y (valor absoluto)

            totalVariation += dx + dy;  // Suma las diferencias absolutas
        }
    }

    return totalVariation;
}

// Contar los valores de intensidad únicos
std::set<int> getUniqueLevels(const cv::Mat& img)
{
    std::set<int> uniqueLevels;
    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            uniqueLevels.insert(img.at<uchar>(y, x));
        }
    }
    return uniqueLevels;
}

cv::Mat drawHist(const cv::Mat& img, int binSize = 3, int height = 300)
{
    if (img.empty() || img.channels() != 1)
    {
        std::cerr << "La imagen debe estar en escala de grises." << std::endl;
        return cv::Mat();
    }

    cv::Mat dst;

    std::vector<int> data(256, 0);
    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            uchar pixelValue = img.at<uchar>(y, x);
            data[pixelValue]++;
        }
    }

    int max_value = *std::max_element(data.begin(), data.end());
    int rows = height;
    int cols = data.size() * binSize;

    dst = cv::Mat(rows, cols, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i < data.size(); ++i)
    {
        int binHeight = static_cast<int>((data[i] / static_cast<float>(max_value)) * rows);
        cv::rectangle(dst,
                      cv::Point(i * binSize, rows - binHeight),
                      cv::Point((i + 1) * binSize - 1, rows),
                      (i % 2) ? cv::Scalar(0, 100, 255) : cv::Scalar(0, 0, 255),
                      cv::FILLED);
    }

    cv::imshow("Histogram", dst);
    cv::waitKeyEx(0);

    return dst;
}

cv::Mat generateGaussianImage(int width, int height)
{
    cv::Mat image(height, width, CV_64FC1);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            double dx = (x - width / 2.0) / (width / 4.0);
            double dy = (y - height / 2.0) / (height / 4.0);
            double value = std::exp(-(dx * dx + dy * dy));
            image.at<double>(y, x) = value;
        }
    }

    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);

    return image;
}

int main()
{
    /**
     * Mostrar una al lado de otra la imagen gris original y sus isofotas (level sets) a la derecha, o al menos algunos de ellos (0-255).
     * Y hacer pequeña prueba para demostrar que en imágenes binarias la minimización de la TV es equivalente a la minimización de la
     * longitud del (único) contorno binario (fórmula de co-área).
     */

    /// Dando valores arbitrarios al nº de level sets para visualizarlos
    {
        // int num_levels = 50;
        // int thickness = 1;
        // const std::string input_image_path = "/media/sf_shared_folder/ImageProcessingSamples/mountain.tif";
        // const std::string save_path = "/opt/proyectos/computerVisionMiscellaneous/2025-XX-XX-level-sets/data/";

        // double totalLength;
        // cv::Mat img = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);
        // cv::resize(img, img, cv::Size(800, 600), 0.0, 0.0, cv::INTER_NEAREST_EXACT);
        // cv::Mat displayImage = showLevelSets(img, totalLength, num_levels, thickness);

        // cv::Mat hist = drawHist(img, 3, 300);

        // std::cout << "Perímetro de los level sets: " << totalLength << std::endl;

        // cv::imwrite(save_path + "hist.png", hist);
        // cv::imwrite(save_path + "original.png", img);
        // cv::imwrite(save_path + "displayImage.png", displayImage);
    }

    /// Encontrando el número exacta de level sets que tiene la imagen
    {
        const std::string input_image_path = "/media/sf_shared_folder/ImageProcessingSamples/kiel.pgm";
        const std::string save_path = "/opt/proyectos/computerVisionMiscellaneous/2025-XX-XX-level-sets/data/";

        double totalLength;
        cv::Mat img = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(800, 600), 0.0, 0.0, cv::INTER_NEAREST_EXACT);

        cv::GaussianBlur(img, img, cv::Size(33, 33), 0.0, 0.0, cv::BORDER_DEFAULT);

        cv::Mat hist = drawHist(img, 3, 300);

        std::set<int> uniqueLevels = getUniqueLevels(img);
        std::cout << "Número de level sets únicos: " << uniqueLevels.size() << std::endl;
        std::cout << "Valores de intensidad presentes: ";
        for (int level : uniqueLevels)
        {
            std::cout << level << " ";
        }
        std::cout << std::endl;

        int thickness = 1;
        std::vector<int> intensityLevels(uniqueLevels.begin(), uniqueLevels.end());

        cv::Mat displayImage = showLevelSets(img, totalLength, intensityLevels, thickness);

        double tv_iso = computeTotalVariationIsotropic(img);
        double tv_anis = computeTotalVariationAnisotropic(img);
        std::cout << "Variación Total iso. de la imagen: " << tv_iso << std::endl;
        std::cout << "Variación Total aniso. de la imagen: " << tv_anis << std::endl;
        std::cout << "Perímetro de los level sets: " << totalLength << std::endl;

        cv::imwrite(save_path + "hist.png", hist);
        cv::imwrite(save_path + "original.png", img);
        cv::imwrite(save_path + "displayImage.png", displayImage);
    }

    /// Lo mismo que antes pero con una imagen de BV sintética
    {
        // const std::string save_path = "/opt/proyectos/computerVisionMiscellaneous/2025-XX-XX-level-sets/data/";

        // double totalLength;
        // cv::Mat img = generateGaussianImage(1920, 1080);
        // img.convertTo(img, CV_8UC1, 255.0);

        // cv::GaussianBlur(img, img, cv::Size(3, 3), 0.0, 0.0, cv::BORDER_DEFAULT);

        // cv::Mat hist = drawHist(img, 3, 300);

        // std::set<int> uniqueLevels = getUniqueLevels(img);
        // std::cout << "Número de level sets únicos: " << uniqueLevels.size() << std::endl;
        // std::cout << "Valores de intensidad presentes: ";
        // for (int level : uniqueLevels)
        // {
        //     std::cout << level << " ";
        // }
        // std::cout << std::endl;

        // int thickness = 1;
        // std::vector<int> intensityLevels(uniqueLevels.begin(), uniqueLevels.end());

        // cv::Mat displayImage = showLevelSets(img, totalLength, intensityLevels, thickness);

        // double tv_iso = computeTotalVariationIsotropic(img);
        // double tv_anis = computeTotalVariationAnisotropic(img);
        // std::cout << "Variación Total iso. de la imagen: " << tv_iso << std::endl;
        // std::cout << "Variación Total aniso. de la imagen: " << tv_anis << std::endl;
        // std::cout << "Perímetro de los level sets: " << totalLength << std::endl;

        // cv::imwrite(save_path + "hist.png", hist);
        // cv::imwrite(save_path + "original.png", img);
        // cv::imwrite(save_path + "displayImage.png", displayImage);
    }

    return 0;
}
