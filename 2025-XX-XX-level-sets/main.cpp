#include <iostream>
#include <random>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>

// Función que genera y muestra los conjuntos de nivel con colores distintos
cv::Mat showLevelSets(const cv::Mat& img, int numLevels, int thickness = 1) {
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
        cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (size_t j = 0; j < contours.size(); ++j) {
            cv::drawContours(displayImage, contours, (int)j, color, thickness, cv::LINE_AA);
        }
    }

    cv::imshow("Original with Level Sets", displayImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return displayImage;
}

int main()
{
    /**
     * Mostrar una al lado de otra la imagen gris original y sus isofotas (level sets) a la derecha, o al menos algunos de ellos (0-255).
     * Y hacer pequeña prueba para demostrar que en imágenes binarias la minimización de la TV es equivalente a la minimización de la
     * longitud del (único) contorno binario (fórmula de co-área).
     */
    {
        int num_levels = 5;
        int thickness = 3;
        const std::string input_image_path = "/media/sf_shared_folder/ImageProcessingSamples/squares.tif";
        const std::string save_path = "/opt/proyectos/computerVisionMiscellaneous/2025-XX-XX-level-sets/data/";

        cv::Mat img = cv::imread(input_image_path, cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(800, 600), 0.0, 0.0, cv::INTER_NEAREST_EXACT);
        cv::Mat displayImage = showLevelSets(img, num_levels, thickness);

        cv::imwrite(save_path + "original.png", img);
        cv::imwrite(save_path + "displayImage.png", displayImage);
    }

    return 0;
}
