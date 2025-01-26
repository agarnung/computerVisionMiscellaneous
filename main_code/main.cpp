/**
 * Varios algoritmos e ideas sobre Procesamiento de Imagen y Visión por Computador recogidos de la literatura
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <x86intrin.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#define EIGEN_USE_BLAS // Usar optimizaciones de BLAS
#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <immintrin.h> // Para AVX y SSE intrínsecos

#include "cvlibrary.h"


/// @param doFiltering [in] true == denoising, false == segmentación
void MeanShiftFilteringAndSegmentation(cv::Mat& Img, int hs, int hr, bool doFiltering)
{
    #define MS_MAX_NUM_CONVERGENCE_STEPS 5
    #define MS_MEAN_SHIFT_TOL_COLOR 0.3
    #define MS_MEAN_SHIFT_TOL_SPATIAL 0.3

    // Estructura para manejar puntos en un espacio de color 5D
    struct Point5D {
        float x, y, l, a, b;

        Point5D() : x(-1), y(-1), l(-1), a(-1), b(-1) {}

        void PointLab()
        {
            l = l * 100 / 255;
            a -= 128;
            b -= 128;
        }

        void PointRGB()
        {
            l = l * 255 / 100;
            a += 128;
            b += 128;
        }

        void Accumulate(const Point5D& Pt) {
            x += Pt.x;
            y += Pt.y;
            l += Pt.l;
            a += Pt.a;
            b += Pt.b;
        }

        void Copy(const Point5D& Pt)
        {
            x = Pt.x;
            y = Pt.y;
            l = Pt.l;
            a = Pt.a;
            b = Pt.b;
        }

        float ColorDistance(const Point5D& Pt) const
        {
            return std::sqrt(std::pow(l - Pt.l, 2) + std::pow(a - Pt.a, 2) + std::pow(b - Pt.b, 2));
        }

        float SpatialDistance(const Point5D& Pt) const
        {
            return std::sqrt(std::pow(x - Pt.x, 2) + std::pow(y - Pt.y, 2));
        }

        void Scale(float scale)
        {
            x *= scale;
            y *= scale;
            l *= scale;
            a *= scale;
            b *= scale;
        }

        void Set(float px, float py, float pl, float pa, float pb)
        {
            x = px; y = py; l = pl; a = pa; b = pb;
        }
    };

    const int dxdy[][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};

    int ROWS = Img.rows;
    int COLS = Img.cols;
    std::vector<cv::Mat> IMGChannels;
    cv::split(Img, IMGChannels);

    Point5D PtCur, PtPrev, PtSum, Pt;
    int Left, Right, Top, Bottom, NumPts, step;

    if (doFiltering) {
        // Filtrado Mean Shift
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                Left = std::max(j - hs, 0);
                Right = std::min(j + hs, COLS);
                Top = std::max(i - hs, 0);
                Bottom = std::min(i + hs, ROWS);
                PtCur.Set(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
                PtCur.PointLab();
                step = 0;

                do {
                    PtPrev.Copy(PtCur);
                    PtSum.Set(0, 0, 0, 0, 0);
                    NumPts = 0;

                    for (int hx = Top; hx < Bottom; hx++) {
                        for (int hy = Left; hy < Right; hy++) {
                            Pt.Set(hx, hy, (float)IMGChannels[0].at<uchar>(hx, hy), (float)IMGChannels[1].at<uchar>(hx, hy), (float)IMGChannels[2].at<uchar>(hx, hy));
                            Pt.PointLab();
                            if (PtCur.ColorDistance(Pt) < hr) {
                                PtSum.Accumulate(Pt);
                                NumPts++;
                            }
                        }
                    }

                    if (NumPts > 0) {
                        PtSum.Scale(1.0 / NumPts);
                        PtCur.Copy(PtSum);
                    }
                    step++;
                } while ((PtCur.ColorDistance(PtPrev) > MS_MEAN_SHIFT_TOL_COLOR) && (PtCur.SpatialDistance(PtPrev) > MS_MEAN_SHIFT_TOL_SPATIAL) && (step < MS_MAX_NUM_CONVERGENCE_STEPS));

                PtCur.PointRGB();
                Img.at<cv::Vec3b>(i, j) = cv::Vec3b(PtCur.l, PtCur.a, PtCur.b);
            }
        }
    } else {
        // Segmentación
        int RegionNumber = 0;
        int label = -1;
        float* Mode = new float[ROWS * COLS * 3]();
        int* MemberModeCount = new int[ROWS * COLS]();
        std::memset(MemberModeCount, 0, ROWS * COLS * sizeof(int));

        int** Labels = new int* [ROWS];
        for (int i = 0; i < ROWS; i++)
            Labels[i] = new int[COLS];

        // Inicialización
        for (int i = 0; i < ROWS; i++)
            std::fill(Labels[i], Labels[i] + COLS, -1);

        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                if (Labels[i][j] < 0) {
                    Labels[i][j] = ++label;
                    PtCur.Set(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
                    PtCur.PointLab();

                    Mode[label * 3 + 0] = PtCur.l;
                    Mode[label * 3 + 1] = PtCur.a;
                    Mode[label * 3 + 2] = PtCur.b;

                    std::vector<Point5D> NeighbourPoints;
                    NeighbourPoints.push_back(PtCur);

                    while (!NeighbourPoints.empty()) {
                        Pt = NeighbourPoints.back();
                        NeighbourPoints.pop_back();

                        for (int k = 0; k < 8; k++) {
                            int hx = Pt.x + dxdy[k][0];
                            int hy = Pt.y + dxdy[k][1];
                            if ((hx >= 0) && (hy >= 0) && (hx < ROWS) && (hy < COLS) && (Labels[hx][hy] < 0)) {
                                Point5D P;
                                P.Set(hx, hy, (float)IMGChannels[0].at<uchar>(hx, hy), (float)IMGChannels[1].at<uchar>(hx, hy), (float)IMGChannels[2].at<uchar>(hx, hy));
                                P.PointLab();

                                if (PtCur.ColorDistance(P) < hr) {
                                    Labels[hx][hy] = label;
                                    NeighbourPoints.push_back(P);
                                    MemberModeCount[label]++;
                                    Mode[label * 3 + 0] += P.l;
                                    Mode[label * 3 + 1] += P.a;
                                    Mode[label * 3 + 2] += P.b;
                                }
                            }
                        }
                    }
                    MemberModeCount[label]++;
                    Mode[label * 3 + 0] /= MemberModeCount[label];
                    Mode[label * 3 + 1] /= MemberModeCount[label];
                    Mode[label * 3 + 2] /= MemberModeCount[label];
                }
            }
        }
        RegionNumber = label + 1;

        // Construcción de la imagen resultante
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                label = Labels[i][j];
                float l = Mode[label * 3 + 0];
                float a = Mode[label * 3 + 1];
                float b = Mode[label * 3 + 2];
                Pt.Set(i, j, l, a, b);
                Pt.PointRGB();
                Img.at<cv::Vec3b>(i, j) = cv::Vec3b(Pt.l, Pt.a, Pt.b);
            }
        }

        // Liberación de memoria
        delete[] Mode;
        delete[] MemberModeCount;
        for (int i = 0; i < ROWS; i++)
            delete[] Labels[i];
        delete[] Labels;
    }
}




cv::Mat shockFilter(const cv::Mat& f, int numIter = 30, float dt = 0.25)
{
    if (f.type() != CV_32FC1)
    {
        std::cerr << "La imagen debe ser CV_32FC1, pero es de tipo " << f.type() << std::endl;
        return f;
    }

    cv::Mat g = f.clone();
    int rows = g.rows;
    int cols = g.cols;

    // Iterar para aplicar el filtro de choque
    for (int iter = 0; iter < numIter; ++iter)
    {
        // Crear una copia de la imagen para procesar sin interferencias
        cv::Mat gNew = g.clone();

        // Iterar sobre cada píxel de la imagen (evitando bordes)
        for (int y = 1; y < rows - 1; ++y)
        {
            // Obtener punteros a las filas actuales en las imágenes g y gNew
            float* gPtr = g.ptr<float>(y);
            float* gNewPtr = gNew.ptr<float>(y);

            // Punteros a las filas anteriores y siguientes para evitar repetición de llamadas
            float* gPtrPrev = g.ptr<float>(y - 1);
            float* gPtrNext = g.ptr<float>(y + 1);

            for (int x = 1; x < cols - 1; ++x)
            {
                // Calcular los gradientes en x e y
                float gx = (gPtr[x + 1] - gPtr[x - 1]) / 2.0f;
                float gy = (gPtrNext[x] - gPtrPrev[x]) / 2.0f;

                // Calcular las segundas derivadas en x y en y (laplaciano)
                float gxx = gPtr[x + 1] - 2 * gPtr[x] + gPtr[x - 1];
                float gyy = gPtrNext[x] - 2 * gPtr[x] + gPtrPrev[x];

                // Calcular la derivada cruzada gxy
                float gxy = (g.ptr<float>(y + 1)[x + 1] - g.ptr<float>(y - 1)[x + 1] -
                             g.ptr<float>(y + 1)[x - 1] + g.ptr<float>(y - 1)[x - 1]) / 4.0f;

                // Calcular la magnitud del gradiente
                float gradMag = std::sqrt(gx * gx + gy * gy);

                // Laplaciano basado en derivadas cruzadas
                float del2g = gx * gx * gxx + 2 * gx * gy * gxy + gy * gy * gyy;

                // Aplicar el filtro de choque
                float dg_by_dt = -std::copysign(1.0f, del2g) * gradMag;
                gNewPtr[x] = gPtr[x] + dg_by_dt * dt;
            }
        }

        g = gNew;

        cvlib::imshowpair("Shock Filter", f, g);
        cv::waitKey(0);
    }

    return g;
}







// Función que genera y muestra los conjuntos de nivel con colores distintos
void showLevelSets(const cv::Mat& img, int numLevels) {
    if (img.empty() || img.channels() != 1) {
        std::cerr << "La imagen debe ser en escala de grises." << std::endl;
        return;
    }

    if (numLevels < 3) {
        std::cerr << "Mínimo 3 level sets" << std::endl;
        return;
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

    for (int i = 0; i < numLevels; ++i) {
        int threshold_value = (i * 255) / (numLevels - 1);

        cv::Scalar color = getUniqueColor();
        std::cout << "Level " << i << ": Threshold value = " << threshold_value
                  << " | Color = (" << color[0] << ", " << color[1] << ", " << color[2] << ")" << std::endl;

        cv::Mat binaryImage;
        cv::threshold(img, binaryImage, threshold_value, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (size_t j = 0; j < contours.size(); ++j) {
            cv::drawContours(displayImage, contours, (int)j, color, 1);
        }
    }

    cv::imshow("Original with Level Sets", displayImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
}


int main()
{
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
    {
//        cv::Mat image = cv::imread("/home/alejandro/Imágenes/cifrasYLetras/K_2.png", cv::IMREAD_GRAYSCALE);
//        image.setTo(0, image < 127);
//        cv::resize(image, image, cv::Size(512, 512), 0.0, 0.0, cv::INTER_LINEAR_EXACT);
//        const double blancos_pct = (double)cv::countNonZero(image) / image.total();
//        blancos_pct >  0.5 ?
//                    cv::threshold(image, image, 127, 255, cv::THRESH_BINARY_INV) :
//                    cv::threshold(image, image, 127, 255, cv::THRESH_BINARY);
//        image.convertTo(image, CV_64FC1, 1.0 / 255.0, 0.0);
//        cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
//        cv::imshow("image", image);
//        cv::waitKey(0);

//        const int totalCols = image.cols;
//        const int totalRows = image.rows;

//        /// Centroide (x̄, ȳ) de la imagen:
//        const double* img_ptr = image.ptr<double>(0);
//        double sumX{0.0}, sumY{0.0}, sumIntensity{0.0};
//        for (int y = 0; y < totalRows; ++y)
//        {
//            for (int x = 0; x < totalCols; ++x)
//            {
//                double intensity = img_ptr[y * totalCols + x];
//                sumX += x * intensity;
//                sumY += y * intensity;
//                sumIntensity += intensity;
//            }
//        }
//        const double centroidX = sumX / sumIntensity;
//        const double centroidY = sumY / sumIntensity;
//        std::cout << "centroid: (" << (int)centroidX << ", " << (int)centroidY << ")" << std::endl;

//        /// Computar la matriz de dispersión de forma M
//        double sum_m_11{0.0}, sum_m_22{0.0}, sum_m_12_m_21{0.0};
//        for (int y = 0; y < totalRows; ++y)
//        {
//            for (int x = 0; x < totalCols; ++x)
//            {
//                double intensity = img_ptr[y * totalCols + x];
//                sum_m_11 += x * x * intensity;
//                sum_m_22 += y * y * intensity;
//                sum_m_12_m_21 += x * y * intensity;
//            }
//        }
//        const double m11 = sum_m_11 / sumIntensity - centroidX * centroidX;
//        const double m22 = sum_m_22 / sumIntensity - centroidY * centroidY;
//        const double m12 = sum_m_12_m_21 / sumIntensity - centroidX * centroidY;
//        const double m21 = m12;
//        std::cout << "m_11: " << m11 << std::endl;
//        std::cout << "m_22: " << m22 << std::endl;
//        std::cout << "m12: " << m12 << std::endl;
//        std::cout << "m21: " << m21 << std::endl;
////        cv::Moments moments = cv::moments(image, true);
////        double centroidX = moments.m10 / moments.m00;
////        double centroidY = moments.m01 / moments.m00;
////        double m11 = moments.mu20 / moments.m00 - centroidX * centroidX;
////        double m22 = moments.mu02 / moments.m00 - centroidY * centroidY;
////        double m_12_m_21 = moments.mu11 / moments.m00 - centroidX * centroidY;

//        cv::Mat M = (cv::Mat_<double>(2, 2) << m11, m12, m21, m22);
//        std::cout << "M:\n" << M << std::endl;

//        /// Alinear los ejes de coordenadas con los autovectores de M
//        /// (trasladar origen al centroide, rotar ejes a los autovectores
//        /// y escalar según el valor de los autovalores)

//        // Cálculo manual de eigenvalues y eigenvectors:
////        double lambda1, lambda2;
////        const double discriminant = m11 * m11 + 4.0 * m12 * m21 - 2.0 * m11 * m22 + m22 * m22;
////        if (discriminant >= 0)
////        {
////            const double sqrt_discriminant = std::sqrt(discriminant);
////            lambda1 = (m11 + m22 + sqrt_discriminant) / 2.0;
////            lambda2 = (m11 + m22 - sqrt_discriminant) / 2.0;
////            std::cout << "lambda1: " << lambda1 << std::endl;
////            std::cout << "lambda2: " << lambda2 << std::endl;
////        }
////        else
////        {
////            double parte_real = (m11 + m22) / 2.0;
////            double parte_imaginaria = std::sqrt(-discriminant) / 2.0;
////            std::cout << "lambda1: " << parte_real << " + " << parte_imaginaria << "i" << std::endl;
////            std::cout << "lambda2: " << parte_real << " - " << parte_imaginaria << "i" << std::endl;
////            std::cout << "autovalores complejos" << std::endl;
////            exit(-1);
////        }
////        /// @see https://www.soest.hawaii.edu/martel/Courses/GG303/Eigenvectors.pdf
////        cv::Vec2d E1(
////            m12 / std::sqrt(m12 * m12 - (lambda1 - m11) * (lambda1 - m11)),
////            m12 / std::sqrt(m12 * m12 - (lambda1 - m22) * (lambda1 - m22))
////        );
////        cv::Vec2d E2(
////            m12 / std::sqrt(m12 * m12 - (lambda2 - m11) * (lambda2 - m11)),
////            m12 / std::sqrt(m12 * m12 - (lambda2 - m22) * (lambda2 - m22))
////        );
////        std::cout << "E1: " << E1 << std::endl;
////        std::cout << "E2: " << E2 << std::endl;
////        E1 /= cv::norm(E1);
////        E2 /= cv::norm(E2);

//        // Cálculo de eigenvalues y eigenvectors con OpenCV
//        cv::Mat eigenvalues, eigenvectors;
//        cv::eigen(M, eigenvalues, eigenvectors);
//        double lambda1 = eigenvalues.at<double>(0, 0);
//        double lambda2 = eigenvalues.at<double>(1, 0);
//        cv::Vec2d E1(eigenvectors.at<double>(0, 0), eigenvectors.at<double>(1, 0));
//        cv::Vec2d E2(eigenvectors.at<double>(0, 1), eigenvectors.at<double>(1, 1));
//        std::cout << "Eigenvalue lambda1: " << lambda1 << std::endl;
//        std::cout << "Eigenvalue lambda2: " << lambda2 << std::endl;
//        std::cout << "Eigenvector E1: " << E1 << std::endl;
//        std::cout << "Eigenvector E2: " << E2 << std::endl;

//        cv::Mat R(2, 2, CV_64FC1);
//        R.at<double>(0, 0) = E1[0];
//        R.at<double>(0, 1) = E1[1];
//        R.at<double>(1, 0) = E2[0];
//        R.at<double>(1, 1) = E2[1];
//        std::cout << "Matriz de rotación R:" << R << std::endl;

//        /// Calcular k dinámicamente para que el área del objeto transformado sea la misma que la original
//        const int whitePixelCountOriginal = cv::countNonZero(image);

//        cv::Mat normalized;

//        double k = 1.0;
//        double tolerance = 0.1; // Tolerancia para el ajuste de k
//        double error = std::numeric_limits<double>::max();
//        int maxIterations = 50;
//        int iteration = 0;
//        while (error > tolerance && iteration < maxIterations)
//        {
//            double minX = std::numeric_limits<double>::max();
//            double minY = std::numeric_limits<double>::max();
//            double maxX = std::numeric_limits<double>::min();
//            double maxY = std::numeric_limits<double>::min();
//            const double* R_ptr = R.ptr<double>(0);

//            for (int y = 0; y < totalRows; ++y)
//            {
//                for (int x = 0; x < totalCols; ++x)
//                {
//                    double x_centered = x - centroidX;
//                    double y_centered = y - centroidY;

//                    double x_rotated = R_ptr[0] * x_centered + R_ptr[1] * y_centered;
//                    double y_rotated = R_ptr[2] * x_centered + R_ptr[3] * y_centered;

//                    double x_scaled = (k / std::sqrt(lambda1)) * x_rotated;
//                    double y_scaled = (k / std::sqrt(lambda2)) * y_rotated;

//                    if (x_scaled < minX) minX = x_scaled;
//                    if (x_scaled > maxX) maxX = x_scaled;
//                    if (y_scaled < minY) minY = y_scaled;
//                    if (y_scaled > maxY) maxY = y_scaled;
//                }
//            }

//            int newCols = static_cast<int>(std::ceil(maxX - minX + 1));
//            int newRows = static_cast<int>(std::ceil(maxY - minY + 1));

//            normalized = cv::Mat(newRows, newCols, CV_64FC1, cv::Scalar(0));
//            double* normalizedPtr = normalized.ptr<double>(0);
//            for (int y = 0; y < totalRows; ++y)
//            {
//                for (int x = 0; x < totalCols; ++x)
//                {
//                    double intensity = img_ptr[y * totalCols + x];

//                    double x_centered = x - centroidX;
//                    double y_centered = y - centroidY;

//                    double x_rotated = R_ptr[0] * x_centered + R_ptr[1] * y_centered;
//                    double y_rotated = R_ptr[2] * x_centered + R_ptr[3] * y_centered;

//                    double x_scaled = (k / std::sqrt(lambda1)) * x_rotated;
//                    double y_scaled = (k / std::sqrt(lambda2)) * y_rotated;

//                    int newX = static_cast<int>(std::round(x_scaled - minX));
//                    int newY = static_cast<int>(std::round(y_scaled - minY));

//                    if (newX >= 0 && newX < newCols && newY >= 0 && newY < newRows)
//                        normalizedPtr[newY * newCols + newX] = intensity;
//                }
//            }

//            const int whitePixelCountTransformed = cv::countNonZero(normalized);

//            std::cout << "whitePixelCountTransformed: " << whitePixelCountTransformed << std::endl;
//            error = std::abs((double)whitePixelCountOriginal - (double)whitePixelCountTransformed) / (double)whitePixelCountOriginal;
//            std::cout << "Iteration: " << iteration << ", Error: " << error << std::endl;

//            if (whitePixelCountTransformed < whitePixelCountOriginal)
//                k *= 1.1;
//            else
//                k *= 0.9;

//            ++iteration;
//        }

//        /// Para refinar el resultado, recortar la bounding box del objeto
//        int filaInicio = 0;
//        int filaFin = normalized.rows - 1;
//        int colInicio = 0;
//        int colFin = normalized.cols - 1;
//        for (int i = 0; i < normalized.rows; ++i)
//        {
//            if (cv::countNonZero(normalized.row(i)) > 0)
//            {
//                filaInicio = i;
//                break;
//            }
//        }
//        for (int i = normalized.rows - 1; i >= filaInicio; --i)
//        {
//            if (cv::countNonZero(normalized.row(i)) > 0)
//            {
//                filaFin = i;
//                break;
//            }
//        }
//        for (int i = 0; i < normalized.cols; ++i)
//        {
//            if (cv::countNonZero(normalized.col(i)) > 0)
//            {
//                colInicio = i;
//                break;
//            }
//        }
//        for (int i = normalized.cols - 1; i >= colInicio; --i)
//        {
//            if (cv::countNonZero(normalized.col(i)) > 0)
//            {
//                colFin = i;
//                break;
//            }
//        }
//        cv::Rect roi(colInicio, filaInicio, colFin - colInicio + 1, filaFin - filaInicio + 1);
//        normalized = normalized(roi).clone();

//        cv::namedWindow("normalized", cv::WINDOW_AUTOSIZE);
//        cv::imshow("normalized", normalized);
//        cv::waitKey(0);
    }





    /**
     * Data-Oriented Programming vs Object-Oriented Programming
     * @see Mike Acton CppCon 2014
     */
    {
//        const int num_entities = 1000000;

//        /// Clase OOP con mal orden de atributos, causando padding
//        /// El padding será usado para alinear correctamente los atributos en memoria
//        /// El orden de los atributos es aleatorio y no toma en cuenta el tamaño de cada tipo de dato.
//        /// Esto causa que el compilador agregue padding (bytes de relleno) para alinear correctamente
//        /// los atributos en la memoria. Como resultado, se desperdicia memoria.
//        std::chrono::duration<double> elapsedOOPBad;
//        class Entity_OOP_Bad {
//        public:
//            struct atributes {
////                bool active;       // 1 byte
////                double dx, dy, dz; // 8 bytes cada uno (24 bytes en total)
////                float x, y, z;     // 4 bytes cada uno (12 bytes)
////                uint16_t something; // 2 bytes
////                char id;           // 1 byte
////                int score;         // 4 bytes
//                double dx, dy, dz; // 8 bytes cada uno (24 bytes en total)
//                float x, y, z;     // 4 bytes cada uno (12 bytes)
//                uint16_t something; // 2 bytes
//                uint16_t something1; // 2 bytes
//                uint16_t something2; // 2 bytes
//                int score;         // 4 bytes
//                int score1;         // 4 bytes
//                int score2;         // 4 bytes
//                char id;           // 1 byte
//                bool active;       // 1 byte
//            };

//            atributes mAtributes;

//            void modifyParams(){
//                this->mAtributes.x = this->mAtributes.y = this->mAtributes.z = 0.0f;
//                this->mAtributes.dx = this->mAtributes.dy = this->mAtributes.dz = 0.1;
//                this->mAtributes.active = true;
//                this->mAtributes.id = 'A';
//                this->mAtributes.score = 100;
//                this->mAtributes.score1 = 100;
//                this->mAtributes.score2 = 100;
//                this->mAtributes.something *= 2;
//                this->mAtributes.something1 *= 2;
//                this->mAtributes.something2 *= 2;
//            }
//        };
//        {
//            std::vector<Entity_OOP_Bad> entities(num_entities);
//            auto start = std::chrono::high_resolution_clock::now();
//            unsigned long long start_cycles = __rdtsc();
//            for (auto& entity : entities) entity.modifyParams();
//            unsigned long long end_cycles = __rdtsc();
//            elapsedOOPBad = std::chrono::high_resolution_clock::now() - start;
//            std::cout << "OOP (Bad Order) CPU cycles: " << (end_cycles - start_cycles) << "\n";
//            std::cout << "OOP (Bad Order) Execution time: " << elapsedOOPBad.count() << " seconds\n";
//        }

//        /// Clase OOP con buen orden de atributos, minimizando el padding
//        /// Aquí se reduce el padding al agrupar los tipos similares
//        /// Aquí los atributos se reorganizan de mayor a menor tamaño (primero double, luego float, después int, char, y finalmente bool).
//        /// Esto minimiza la cantidad de padding necesario, haciendo que la estructura sea más compacta en memoria.
//        /// A un nivel más técnico, al realizar operaciones en los atributos, el código máquina hará búsquedas de registros a partir
//        /// del rax (rax+4, rax+20...) con menos desplazamientos, y por tanto más eficientemente, si los atributos están bien ordenados
//        std::chrono::duration<double> elapsedOOPDOP;
//        class Entity_OOP_Good {
//        public:
//            struct atributes {
//                double dx, dy, dz; // 8 bytes cada uno (24 bytes en total)
//                float x, y, z;     // 4 bytes cada uno (12 bytes)
//                int score;         // 4 bytes
//                int score1;         // 4 bytes
//                int score2;         // 4 bytes
//                uint16_t something; // 2 bytes
//                uint16_t something1; // 2 bytes
//                uint16_t something2; // 2 bytes
//                char id;           // 1 byte
//                bool active;       // 1 byte
//            };

//            atributes mAtributes;

//            void modifyParams(){
//                this->mAtributes.x = this->mAtributes.y = this->mAtributes.z = 0.0f;
//                this->mAtributes.dx = this->mAtributes.dy = this->mAtributes.dz = 0.1;
//                this->mAtributes.active = true;
//                this->mAtributes.id = 'A';
//                this->mAtributes.score = 100;
//                this->mAtributes.score1 = 100;
//                this->mAtributes.score2 = 100;
//                this->mAtributes.something *= 2;
//                this->mAtributes.something1 *= 2;
//                this->mAtributes.something2 *= 2;
//            }
//        };
//        {
//            std::vector<Entity_OOP_Good> entities(num_entities);
//            auto start = std::chrono::high_resolution_clock::now();
//            unsigned long long start_cycles = __rdtsc();
//            for (auto& entity : entities) entity.modifyParams();
//            unsigned long long end_cycles = __rdtsc();
//            elapsedOOPDOP = std::chrono::high_resolution_clock::now() - start;
//            std::cout << "OOP (Good Order by DOP) CPU cycles: " << (end_cycles - start_cycles) << "\n";
//            std::cout << "OOP (Good Order by DOP) Execution time: " << elapsedOOPDOP.count() << " seconds\n";
//        }

//        /// Con el comando $ lscpu se puede ver la información de mi CPU, para ver el tamaño en bytes que la CPU consulta en cada ciclo, para
//        /// saber cómo maximizar la eficiencia de mi estructura para que no haya huecos innecesarios y hacer las operaciones en el menos número
//        /// de ciclos (tamaños cachés L1 y L2, tamaño del bus de datos de 64 bits...)
//        ///
//        std::chrono::duration<double> elapsedOOPDOP_GoodWithFooPadding;
//        class Entity_OOP_GoodWithFooPadding {
//        public:
//            struct atributes {
//                double dx, dy, dz;  // 8 bytes cada uno (24 bytes en total)
//                float x, y, z;      // 4 bytes cada uno (12 bytes)
//                int score;         // 4 bytes
//                int score1;         // 4 bytes
//                int score2;         // 4 bytes
//                uint16_t something; // 2 bytes
//                uint16_t something1; // 2 bytes
//                uint16_t something2; // 2 bytes
//                char id;            // 1 byte
//                bool active;        // 1 byte
//                char padding[8];   // 8 bytes de padding para completar el bloque de 64 bytes
//            };

//            atributes mAtributes;

//            void modifyParams() {
//                this->mAtributes.x = this->mAtributes.y = this->mAtributes.z = 0.0f;
//                this->mAtributes.dx = this->mAtributes.dy = this->mAtributes.dz = 0.1;
//                this->mAtributes.active = true;
//                this->mAtributes.id = 'A';
//                this->mAtributes.score = 100;
//                this->mAtributes.score1 = 100;
//                this->mAtributes.score2 = 100;
//                this->mAtributes.something *= 2;
//                this->mAtributes.something1 *= 2;
//                this->mAtributes.something2 *= 2;
//            }
//        };
//        {
//            std::vector<Entity_OOP_GoodWithFooPadding> entities(num_entities);
//            auto start = std::chrono::high_resolution_clock::now();
//            unsigned long long start_cycles = __rdtsc();
//            for (auto& entity : entities) entity.modifyParams();
//            unsigned long long end_cycles = __rdtsc();
//            elapsedOOPDOP_GoodWithFooPadding = std::chrono::high_resolution_clock::now() - start;
//            std::cout << "OOP (Good Order by DOP and Foo Padding) CPU cycles: " << (end_cycles - start_cycles) << "\n";
//            std::cout << "OOP (Good Order by DOP and Foo Padding) Execution time: " << elapsedOOPDOP_GoodWithFooPadding.count() << " seconds\n";
//        }

//        std::cout << "With DOP is " << (elapsedOOPBad.count() - elapsedOOPDOP.count()) * 1e3 << " ms faster\n";
//        std::cout << "With DOP and Foo Padding is " << (elapsedOOPBad.count() - elapsedOOPDOP_GoodWithFooPadding.count()) * 1e3 << " ms faster\n";

//        /// Uso de RDTSC para contar ciclos de CPU
//        {
////            const int num_entities = 1000;
////            std::vector<float> data(num_entities, 1.0f);
////            unsigned long long start_cycles = __rdtsc();
////            for (int i = 0; i < num_entities; ++i) {
////                data[i] *= 1.1f;
////            }
////            unsigned long long end_cycles = __rdtsc();
////            std::cout << "CPU cycles: " << (end_cycles - start_cycles) << "\n";
//        }

//        /// Las CPU modernas acceden a la memoria en bloques (típicamente de 8 bytes o más). Si los datos están correctamente alineados
//        /// en la memoria, el acceso es más rápido porque puede cargar y almacenar los datos en un solo ciclo de memoria
//        /// Si los datos no están alineados correctamente, la CPU puede tener que realizar más accesos a memoria, lo que introduce
//        /// penalizaciones de rendimiento debido a la necesidad de corregir la alineación en tiempo de ejecución.
    }





    /**
     * Buscar la forma más rápida de transformar una nube de puntos (metiante OpenCV, Eigen, PCL, GLM, en crudo, en paralelo...?
     */
    {
//        auto measure_time = [=](const std::function<void()>& func) {
//            auto start = std::chrono::high_resolution_clock::now();
//            func();
//            auto end = std::chrono::high_resolution_clock::now();
//            return std::chrono::duration<double>(end - start).count();
//        };

//        // Aplicar transformación usando OpenCV directamente
//        auto transform_opencv = [=](const cv::Mat& cloud, const cv::Mat& transform, cv::Mat& result) {
//            result = transform * cloud;
//        };

//        // Aplicar transformación usando OpenCV manualmente con punteros
//        auto transform_opencv_pointers_double = [=](const cv::Mat& cloud, const cv::Mat& transform, cv::Mat& result) {
//            const double* cloudPtr = cloud.ptr<double>(0);
//            const double* transformPtr = transform.ptr<double>(0);
//            double* resultPtr = result.ptr<double>(0);
//            for (int i = 0; i < cloud.cols; ++i)
//            {
//                double x = cloudPtr[i * 3 + 0];
//                double y = cloudPtr[i * 3 + 1];
//                double z = cloudPtr[i * 3 + 2];

//                resultPtr[i * 3 + 0] = transformPtr[0] * x + transformPtr[1] * y + transformPtr[2] * z + transformPtr[3];
//                resultPtr[i * 3 + 1] = transformPtr[4] * x + transformPtr[5] * y + transformPtr[6] * z + transformPtr[7];
//                resultPtr[i * 3 + 2] = transformPtr[8] * x + transformPtr[9] * y + transformPtr[10] * z + transformPtr[11];
//            }
//        };

//        // Aplicar transformación usando OpenCV manualmente con Vec3d
//        auto transform_opencv_pointers_Vec3d = [=](const cv::Mat& cloud, const cv::Mat& transform, cv::Mat& result) {
//            const cv::Vec3d* cloudPtr = cloud.ptr<cv::Vec3d>(0);
//            const double* transformPtr = transform.ptr<double>(0);
//            cv::Vec3d* resultPtr = result.ptr<cv::Vec3d>(0);
//            for (int i = 0; i < cloud.cols; ++i)
//            {
//                const cv::Vec3d& point = cloudPtr[i];

//                double x = point[0];
//                double y = point[1];
//                double z = point[2];

//                resultPtr[i][0] = transformPtr[0] * x + transformPtr[1] * y + transformPtr[2] * z + transformPtr[3];
//                resultPtr[i][1] = transformPtr[4] * x + transformPtr[5] * y + transformPtr[6] * z + transformPtr[7];
//                resultPtr[i][2] = transformPtr[8] * x + transformPtr[9] * y + transformPtr[10] * z + transformPtr[11];
//            }
//        };

//        // Aplicar transformación usando Eigen
//        auto transform_eigen = [=](const Eigen::MatrixXd& cloud, const Eigen::Matrix4d& transform, Eigen::MatrixXd& result) {
//            result = transform * cloud;
//        };

//        // Aplicar transformación en paralelo
//        auto transform_parallel = [=](const cv::Mat& cloud, const cv::Mat& transform, cv::Mat& result) {
//            const double* cloudPtr = cloud.ptr<double>(0);
//            const double* transformPtr = transform.ptr<double>(0);
//            double* resultPtr = result.ptr<double>(0);
//            #pragma omp parallel for
//            for (int i = 0; i < cloud.cols; ++i)
//            {
//                double x = cloudPtr[i * 3 + 0];
//                double y = cloudPtr[i * 3 + 1];
//                double z = cloudPtr[i * 3 + 2];

//                resultPtr[i * 3 + 0] = transformPtr[0] * x + transformPtr[1] * y + transformPtr[2] * z + transformPtr[3];
//                resultPtr[i * 3 + 1] = transformPtr[4] * x + transformPtr[5] * y + transformPtr[6] * z + transformPtr[7];
//                resultPtr[i * 3 + 2] = transformPtr[8] * x + transformPtr[9] * y + transformPtr[10] * z + transformPtr[11];
//            }
//        };

//        // Aplicar transformación con CUDA
//        auto transform_with_cuda = [=](const float* d_cloud, const float* d_transform, float* d_result, int num_points) {
//            //...
//        };

//        // Kernel SIMD para transformar la nube de puntos
//        auto transformWithSIMD = [=](const double* cloud, const double* transform, double* result, int num_points) {
//            int num_simd_points = (num_points / 4) * 4;  // Puntos manejados con SIMD, múltiplo de 4

//            // Cargar las filas de la matriz de transformación en registros SIMD
//            __m256d t0 = _mm256_set1_pd(transform[0]);
//            __m256d t1 = _mm256_set1_pd(transform[1]);
//            __m256d t2 = _mm256_set1_pd(transform[2]);
//            __m256d t3 = _mm256_set1_pd(transform[3]);

//            __m256d t4 = _mm256_set1_pd(transform[4]);
//            __m256d t5 = _mm256_set1_pd(transform[5]);
//            __m256d t6 = _mm256_set1_pd(transform[6]);
//            __m256d t7 = _mm256_set1_pd(transform[7]);

//            __m256d t8 = _mm256_set1_pd(transform[8]);
//            __m256d t9 = _mm256_set1_pd(transform[9]);
//            __m256d t10 = _mm256_set1_pd(transform[10]);
//            __m256d t11 = _mm256_set1_pd(transform[11]);

//            // Procesar bloques de 4 puntos a la vez
//            for (int i = 0; i < num_simd_points; i += 4) {
//                // Cargar las coordenadas X, Y y Z de 4 puntos
//                __m256d x = _mm256_set_pd(cloud[(i + 3) * 3], cloud[(i + 2) * 3], cloud[(i + 1) * 3], cloud[i * 3]);
//                __m256d y = _mm256_set_pd(cloud[(i + 3) * 3 + 1], cloud[(i + 2) * 3 + 1], cloud[(i + 1) * 3 + 1], cloud[i * 3 + 1]);
//                __m256d z = _mm256_set_pd(cloud[(i + 3) * 3 + 2], cloud[(i + 2) * 3 + 2], cloud[(i + 1) * 3 + 2], cloud[i * 3 + 2]);

//                // Realizar la transformación
//                __m256d res_x = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(t0, x), _mm256_mul_pd(t1, y)),
//                                              _mm256_add_pd(_mm256_mul_pd(t2, z), t3));
//                __m256d res_y = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(t4, x), _mm256_mul_pd(t5, y)),
//                                              _mm256_add_pd(_mm256_mul_pd(t6, z), t7));
//                __m256d res_z = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(t8, x), _mm256_mul_pd(t9, y)),
//                                              _mm256_add_pd(_mm256_mul_pd(t10, z), t11));

//                // Almacenar los resultados intercalados (X1, Y1, Z1, X2, Y2, Z2, ...)
//                double temp_x[4], temp_y[4], temp_z[4];
//                _mm256_store_pd(temp_x, res_x);
//                _mm256_store_pd(temp_y, res_y);
//                _mm256_store_pd(temp_z, res_z);

//                for (int j = 0; j < 4; ++j) {
//                    result[(i + j) * 3]     = temp_x[j];
//                    result[(i + j) * 3 + 1] = temp_y[j];
//                    result[(i + j) * 3 + 2] = temp_z[j];
//                }
//            }

//            // Manejar los puntos restantes (si num_points no es múltiplo de 4)
//            for (int i = num_simd_points; i < num_points; ++i) {
//                int base = i * 3;
//                double x = cloud[base];
//                double y = cloud[base + 1];
//                double z = cloud[base + 2];

//                result[base]     = transform[0] * x + transform[1] * y + transform[2] * z + transform[3];
//                result[base + 1] = transform[4] * x + transform[5] * y + transform[6] * z + transform[7];
//                result[base + 2] = transform[8] * x + transform[9] * y + transform[10] * z + transform[11];
//            }
//        };

//        // Crear nube de puntos como una sola fila
//        int num_points = (int)1e6;
//        cv::Mat cloud(1, num_points, CV_64FC3);
//        cv::Mat transform(4, 4, CV_64F);

//        // Inicializar nube de puntos y matriz de transformación
//        cv::randu(cloud, cv::Scalar(0.0, 0.0, 0.0), cv::Scalar(100.0, 100.0, 100.0));

//        cv::Mat cloud_homogeneous(4, cloud.cols, CV_64FC1);
//        for (int i = 0; i < cloud.cols; ++i)
//        {
//            const cv::Vec3d& pt = cloud.at<cv::Vec3d>(0, i);
//            cloud_homogeneous.at<double>(0, i) = pt[0];
//            cloud_homogeneous.at<double>(1, i) = pt[1];
//            cloud_homogeneous.at<double>(2, i) = pt[2];
//            cloud_homogeneous.at<double>(3, i) = 1.0;
//        }

//        transform = cv::Mat::eye(4, 4, CV_64F);
//        {
//            double roll = M_PI / 6;   // Rotación alrededor del eje X
//            double pitch = M_PI / 4;  // Rotación alrededor del eje Y
//            double yaw = M_PI / 3;    // Rotación alrededor del eje Z
//            cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
//                1, 0, 0,
//                0, std::cos(roll), -std::sin(roll),
//                0, std::sin(roll), std::cos(roll));
//            cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
//                std::cos(pitch), 0, std::sin(pitch),
//                0, 1, 0,
//                -std::sin(pitch), 0, std::cos(pitch));
//            cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
//                std::cos(yaw), -std::sin(yaw), 0,
//                std::sin(yaw), std::cos(yaw), 0,
//                0, 0, 1);
//            cv::Mat R = Rz * Ry * Rx;
//            R.copyTo(transform(cv::Rect(0, 0, 3, 3)));
//            transform.at<double>(0, 3) = 10; // Traslación en x
//            transform.at<double>(1, 3) = -5; // Traslación en y
//            transform.at<double>(2, 3) = 3;  // Traslación en z
//        }

//        // Convertir a Eigen => representación matricial
//        Eigen::MatrixXd cloud_eigen(4, num_points);
//        Eigen::Matrix4d transformation_eigen;
//        for (int i = 0; i < num_points; ++i) {
//            cloud_eigen(0, i) = cloud.at<double>(0, i * 3 + 0);
//            cloud_eigen(1, i) = cloud.at<double>(0, i * 3 + 1);
//            cloud_eigen(2, i) = cloud.at<double>(0, i * 3 + 2);
//            cloud_eigen(3, i) = 1.0;
//        }

//        transformation_eigen << transform.at<double>(0, 0), transform.at<double>(0, 1), transform.at<double>(0, 2), transform.at<double>(0, 3),
//                                transform.at<double>(1, 0), transform.at<double>(1, 1), transform.at<double>(1, 2), transform.at<double>(1, 3),
//                                transform.at<double>(2, 0), transform.at<double>(2, 1), transform.at<double>(2, 2), transform.at<double>(2, 3),
//                                transform.at<double>(3, 0), transform.at<double>(3, 1), transform.at<double>(3, 2), transform.at<double>(3, 3);

//        // OpenCV directo
//        cv::Mat result_opencv;
//        result_opencv.create(cloud_homogeneous.rows, cloud_homogeneous.cols, CV_64FC1);
//        double time_opencv = measure_time([&]() {
//            transform_opencv(cloud_homogeneous, transform, result_opencv);
//        });

//        // OpenCV puntero double
//        cv::Mat result_opencv_pt_d;
//        result_opencv_pt_d.create(cloud.rows, cloud.cols, CV_64FC3);
//        double time_opencv_pt_d = measure_time([&]() {
//            transform_opencv_pointers_double(cloud, transform, result_opencv_pt_d);
//        });

//        // OpenCV puntero Vec3d
//        cv::Mat result_opencv_pt_V;
//        result_opencv_pt_V.create(cloud.rows, cloud.cols, CV_64FC3);
//        double time_opencv_pt_V = measure_time([&]() {
//            transform_opencv_pointers_Vec3d(cloud, transform, result_opencv_pt_V);
//        });

//        // Eigen
//        Eigen::MatrixXd result_eigen(4, num_points);
//        double time_eigen = measure_time([&]() {
//            transform_eigen(cloud_eigen, transformation_eigen, result_eigen);
//        });

//        // Paralelo
//        cv::Mat result_parallel;
//        result_parallel.create(cloud.rows, cloud.cols, CV_64FC3);
//        double time_parallel = measure_time([&]() {
//            transform_parallel(cloud, transform, result_parallel);
//        });

//        // SIMD
//        cv::Mat result_SIMD;
//        result_SIMD.create(cloud.rows, cloud.cols, CV_64FC3);

//        double time_SIMD = measure_time([&]() {
//            transformWithSIMD(cloud.ptr<double>(), transform.ptr<double>(), result_SIMD.ptr<double>(), num_points);
//        });

//        // Imprimir por rapidez descendente
//        std::vector<std::pair<std::string, double>> times = {
//            {"OpenCV directo", time_opencv},
//            {"OpenCV puntero double", time_opencv_pt_d},
//            {"OpenCV puntero Vec3d", time_opencv_pt_V},
//            {"Eigen", time_eigen},
//            {"Paralelo", time_parallel},
//            {"SIMD", time_SIMD}
//        };
//        std::sort(times.begin(), times.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
//            return a.second < b.second; // Ordenar de mayor a menor
//        });
//        std::cout << "\n";
//        for (const auto& pair : times)
//            std::cout << "Tiempo " << pair.first << ": " << pair.second << " segundos" << std::endl;

//        auto compare_points = [=](const cv::Vec3d& p1, const cv::Vec3d& p2, double tolerance = 1e-3) {
//            return std::fabs(p1[0] - p2[0]) < tolerance &&
//                   std::fabs(p1[1] - p2[1]) < tolerance &&
//                   std::fabs(p1[2] - p2[2]) < tolerance;
//        };
//        std::cout << "\nSe imprimirá error si hay puntos diferentes..." << std::endl;
//        for (int i = 0; i < num_points; ++i)
//        {
//            cv::Vec3d point_opencv = cv::Vec3d(result_opencv.at<double>(0, i), result_opencv.at<double>(1, i), result_opencv.at<double>(2, i));
//            cv::Vec3d point_opencv_pt_d = result_opencv_pt_d.at<cv::Vec3d>(0, i);
//            cv::Vec3d point_opencv_pt_V = result_opencv_pt_V.at<cv::Vec3d>(0, i);
//            Eigen::Vector4d point_eigen = result_eigen.col(i);
//            cv::Vec3d point_parallel = result_parallel.at<cv::Vec3d>(0, i);
//            cv::Vec3d point_SIMD = result_SIMD.at<cv::Vec3d>(0, i);

//            if (!compare_points(point_opencv, point_opencv_pt_d) || !compare_points(point_opencv, point_opencv_pt_V) ||
//                !compare_points(point_opencv, cv::Vec3d(point_eigen(0), point_eigen(1), point_eigen(2))) ||
//                !compare_points(point_opencv, point_parallel) || !compare_points(point_opencv, point_SIMD))
//            {
//                std::cerr << "Error en el punto " << i << " entre los diferentes métodos." << std::endl;
//                std::cerr << "OpenCV directo: (" << result_opencv.at<double>(0, i) << ", " << result_opencv.at<double>(1, i) << ", " << result_opencv.at<double>(2, i) << ")" << std::endl;
//                std::cerr << "OpenCV puntero double: (" << point_opencv_pt_d[0] << ", " << point_opencv_pt_d[1] << ", " << point_opencv_pt_d[2] << ")" << std::endl;
//                std::cerr << "OpenCV puntero Vec3d: (" << point_opencv_pt_V[0] << ", " << point_opencv_pt_V[1] << ", " << point_opencv_pt_V[2] << ")" << std::endl;
//                std::cerr << "Eigen: (" << point_eigen(0) << ", " << point_eigen(1) << ", " << point_eigen(2) << ")" << std::endl;
//                std::cerr << "Paralelo: (" << point_parallel[0] << ", " << point_parallel[1] << ", " << point_parallel[2] << ")" << std::endl;
//                std::cerr << "SIMD: (" << point_SIMD[0] << ", " << point_SIMD[1] << ", " << point_SIMD[2] << ")" << std::endl;
//            }
//        }
//        int pointToPick{50};
//        cv::Vec3d point_opencv = cv::Vec3d(result_opencv.at<double>(0, pointToPick), result_opencv.at<double>(1, pointToPick), result_opencv.at<double>(2, pointToPick));
//        cv::Vec3d point_opencv_pt_d = result_opencv_pt_d.at<cv::Vec3d>(0, pointToPick);
//        cv::Vec3d point_opencv_pt_V = result_opencv_pt_V.at<cv::Vec3d>(0, pointToPick);
//        Eigen::Vector4d point_eigen = result_eigen.col(pointToPick);
//        cv::Vec3d point_parallel = result_parallel.at<cv::Vec3d>(0, pointToPick);
//        cv::Vec3d point_SIMD = result_SIMD.at<cv::Vec3d>(0, pointToPick);
//        std::cout << "\nPunto " << pointToPick << " entre los diferentes métodos." << std::endl;
//        std::cout << "OpenCV directo: (" << result_opencv.at<double>(0, pointToPick) << ", " << result_opencv.at<double>(1, pointToPick) << ", " << result_opencv.at<double>(2, pointToPick) << ")" << std::endl;
//        std::cout << "OpenCV puntero double: (" << point_opencv_pt_d[0] << ", " << point_opencv_pt_d[1] << ", " << point_opencv_pt_d[2] << ")" << std::endl;
//        std::cout << "OpenCV puntero Vec3d: (" << point_opencv_pt_V[0] << ", " << point_opencv_pt_V[1] << ", " << point_opencv_pt_V[2] << ")" << std::endl;
//        std::cout << "Eigen: (" << point_eigen(0) << ", " << point_eigen(1) << ", " << point_eigen(2) << ")" << std::endl;
//        std::cout << "Paralelo: (" << point_parallel[0] << ", " << point_parallel[1] << ", " << point_parallel[2] << ")" << std::endl;
//        std::cout << "SIMD: (" << point_SIMD[0] << ", " << point_SIMD[1] << ", " << point_SIMD[2] << ")" << std::endl;
    }






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







    /**
     * Shock Filters (sharpening == edge enhancement)
     *
     * Este filtro aplica una evolución basada en PDE hiperbólica (como las de onda) para sharpening, i.e. potenciación de los bordes
     *
     *      df/dt = -sign(delta(f)) * norm(grad(f)),
     *
     * donde:
     * - grad(f) = [fx, fy]^T representa el gradiente de la imagen.
     * - norm(grad(f)) = sqrt(fx^2 + fy^2) es la magnitud del gradiente.
     * - delta(f) es una aproximación del Laplaciano basada en derivadas cruzadas.
     *
     * El filtro de choque se aplica iterativamente, aumentando la nitidez en las zonas de borde y suavizando las regiones planas.
     *
     * @see S. Osher and L.I. Rudin. Feature-oriented image enhancement using shock ﬁlters. SIAM Journal of Numerical Analysis, 27(4):919–940, August 1990.
     * @see Mathematical Image Processing pág. 141
     * @see Jia, "Two-Phase Kernel Estimation for Robust Motion Deblurring", 2013.
     * @see https://github.com/apvijay/shock_filter/blob/master/shock_filter.m
     */
    {
//        cv::Mat f = cv::imread("/home/alejandro/Imágenes/blurry_images/blurryBuilding.png", cv::IMREAD_GRAYSCALE);
//        f.convertTo(f, CV_32FC1, 1.0 / 255.0, 0.0);
//        cv::resize(f , f, cv::Size(512, 512), 0, 0, cv::INTER_NEAREST_EXACT);
////        cv::GaussianBlur(f, f, cv::Size(9, 9), 0);
//        cv::Mat sharpened = shockFilter(f, 30, 0.1);
    }







    /**
     * Mostrar una al lado de otra la imagen gris original y sus isofotas (level sets) a la derecha, o al menos algunos de ellos (0-255).
     * Y hacer pequeña prueba para demostrar que en imágenes binarias la minimización de la TV es equivalente a la minimización de la
     * longitud del (único) contorno binario (fórmula de co-área).
     */
    {
        cv::Mat img = cv::imread("/home/alejandro/Imágenes/ImageProcessingSamples/peppers3.tif", cv::IMREAD_GRAYSCALE);
        cv::resize(img, img, cv::Size(800, 800), 0.0, 0.0, cv::INTER_NEAREST_EXACT);
        int num_levels = 15;
        showLevelSets(img, num_levels);
    }







    /**
     * Estudiar la mejor manera de graficar con C++ una señal 1D, p. ej. cvlib::plotSignal1DGnuplot(g.row(g.rows - 100), "Final");
     */
    {

    }









    /**
     * A. Kheradmand and P. Milanfar. A general framework for regularized, similarity-based image restoration. IEEE Transactions on Image Processing, 23(12):5136–5151, 2014.
     * @brief El paper trata de la restauración de imágenes. Su idea es ver la imagen como un grafo ponderado; la estructura de la imagen se recupera basándose en similitud de
     *        kernel y la graph Laplacian normalizada, la cual evoca un nuevo data term y regularization term. Los coeficientes usados para describir la matriz se hacen de
     *        manera beneficiosa y por tanto la matriz es simétrica, definida positiva y otros atributos buenos para la convexidad/minimización del funcional.
     *        Se resuelve la optimización mediante gradientes conjugados y en cada iteración se recomputan los pesos de la matriz Laplaciana.
     *        La forma del funcional propuesto permite además analizar el espectro de la PDE a través de los valores propios de la Laplaciana.
     *        El método permite distintos tipos de restauraciones, e.g. deblurring, denoising y sharpening.
     */
    {

    }






    /**
     * Modelo generativo para clasificar pixel/vecindad como ruido o defecto (pag 64 simon prince)
     */
    {

    }





    /**
     * Spatial noise filter mediante Suavizado Direccional
     * @see https://arxiv.org/pdf/1608.01993
     * @see pág. 109 image processing and its applications
     */
    {

    }







    /**
     * El sistema de la transf-Z es el modelo de degradación. Buscar los parámetros (el shifting de la respuesta impulsional) que lo deja estable
     * @see Pratts, Digital Image Processing sec 12.2/3/4...
     */
    {

    }





    /**
     * Hacer un solo proyecto probando diferentes operaciones, bien organizado. Kernels de CUDA, funciones com SIMD o AVX, documentar diferencias.
     * Ser estricto al sincronizar hilos con OMP... Todo para poder procesar eficientemente nubes de puntos.
     */
    {

    }






    /**
     * Estudiar los adaptive manifold, qués son para qué sirven https://github.com/znah/notebooks/blob/master/adaptive_manifolds.ipynb
     */
    {

    }
















    /**
     * Sobre ridgelets:
     * @see https://github.com/ElsevierSoftwareX/SOFTX_2019_182/blob/master/ridgelet.m TODO
     * @see https://docs.opencv.org/4.x/d4/d36/classcv_1_1ximgproc_1_1RidgeDetectionFilter.html
     */
    {

    }









    /**
     * Deblurring, traducir metodo graph based!!! https://github.com/BYchao100/Graph-Based-Blind-Image-Deblurring
     */
    {

    }






    /**
     * Traducir split bregman a c++ y escribir teoría en variacional
     * @see https://github.com/ycynyu007/L1-norm-total-variation-inpainting-using-Split_Bregman-Iteration
     */
    {

    }








    /**
     * Plantear un sistema lineal Ax=b cualquiera enfocado en PdI y resolverlo mediante distintos solvers, Jacobi,
     * Gauss-Seidel/Gauss-Jacobi, SOR metod, Primal-Dual... y comparar tiempos y precisión
     * @see file:///C:/Users/Alejandro/Desktop/copy_seg/Daniel%20Cremers%20Variational%20Methods/3_variational_methods3_Variational%20Calculus.pdf
     * @see https://arxiv.org/pdf/2103.16337 secc. 2.3
     */
    {

    }






    /**
     * Que usuarioe especifique matriz A, vector b y escalar g y dibujar algunos los isocontornos en pantalla, y mostrar la
     * forma cuadrática dada
     * @see painless-conjugate-gradient.pdf
     */
    {

    }






    /**
     * Probar implementaciones de TGV y primal dual y apuntar teoría
     * @see https://github.com/yuki-inaho/total_generalized_variation_optimization_python
     */
    {

    }






    /**
     * Crear sinusoide o funcion escalon sintetica corrupta con ruido, p ej xeR^2000, y hacer el denoising con L2 y TV tal como explica boyd
     * y graficarlo, quizas meter un slider para mostrar valor de lambda y el tradeoff entre l2 y tv.
     * @see Apartado 6 aprroximation and fitting, quadratic smoothing

     */
    {

    }






    /**
     * Tone management for photographic look Bae, Paris, Durand [2006]
     * @see https://github.com/alejandromunoznavarro/TFG
     */
    {

    }





    /**
     * Programar el seamless stitching que muestra misha kazhdan, que es una optimización con un regularizador en el
     * gradient vector field (GVF), haciéndolo cero en las costuras y evolucionando el modelo
     * @see min 14 https://www.youtube.com/watch?v=dXytHxl-HdU
     */
    {

    }







    /**
     * Programar también el enhancing que muestra misha kazhdan en el que usa un modelo variacional para escalar los gradientes por un factor.
     * @see min 15:09 https://www.youtube.com/watch?v=dXytHxl-HdU
     */
    {

    }













    /**
     * Mapa de saliencia EM + GMM + Static saliency residual
     * @see https://github.com/SrikanthAmudala/GaussainDistribution/blob/master/algorithm/gmm_smsi.py
     * @see https://github.com/OlaPietka/Applied-Machine-Learning/blob/main/Image%20segmentation%20using%20EM.ipynb
     * @see https://github.com/alessandro-gentilini/gaussianmixturemodel
     * @see https://dennishnf.com/posts/technical/2015-02_expectation-maximization_em_algorithm_in_cpp_using_opencv_2-4/page.html
     * @see file:///home/alejandro/Descargas/Feng_2019_37-1_ifsUnderwatersalientobjectdetectionjointlyusingimprovedspectralresidualandFuzzyc-Means.pdf
     * @see https://github.com/uoip/SpectralResidualSaliency/blob/master/src/saliency.cpp
     * @see https://vgg.fiit.stuba.sk/2015-02/saliency-map/ (más sencillo)
     */
    {

    }






    /**
     * Usar plug and play ADMM y Consensus Equilibrium (CE)
     * @see https://arxiv.org/pdf/1705.08983
     * @see https://github.com/gbuzzard/PnP-MACE
     */
    {

    }





    return 0;
}
