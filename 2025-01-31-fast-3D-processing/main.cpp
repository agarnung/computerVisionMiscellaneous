#include <iostream>
#include <chrono>
#include <vector>
#include <x86intrin.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#define EIGEN_USE_BLAS // Usar optimizaciones de BLAS
#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <immintrin.h> // Para AVX y SSE intrínsecos

int main()
{
    /**
     * Buscar la forma más rápida de transformar una nube de puntos (metiante OpenCV, Eigen, PCL, GLM, en crudo, en paralelo...?
     */
    {
        auto measure_time = [=](const std::function<void()>& func) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double>(end - start).count();
        };

        // Aplicar transformación usando OpenCV directamente
        auto transform_opencv = [=](const cv::Mat& cloud, const cv::Mat& transform, cv::Mat& result) {
            result = transform * cloud;
        };

        // Aplicar transformación usando OpenCV manualmente con punteros
        auto transform_opencv_pointers_double = [=](const cv::Mat& cloud, const cv::Mat& transform, cv::Mat& result) {
            const double* cloudPtr = cloud.ptr<double>(0);
            const double* transformPtr = transform.ptr<double>(0);
            double* resultPtr = result.ptr<double>(0);
            for (int i = 0; i < cloud.cols; ++i)
            {
                double x = cloudPtr[i * 3 + 0];
                double y = cloudPtr[i * 3 + 1];
                double z = cloudPtr[i * 3 + 2];

                resultPtr[i * 3 + 0] = transformPtr[0] * x + transformPtr[1] * y + transformPtr[2] * z + transformPtr[3];
                resultPtr[i * 3 + 1] = transformPtr[4] * x + transformPtr[5] * y + transformPtr[6] * z + transformPtr[7];
                resultPtr[i * 3 + 2] = transformPtr[8] * x + transformPtr[9] * y + transformPtr[10] * z + transformPtr[11];
            }
        };

        // Aplicar transformación usando OpenCV manualmente con Vec3d
        auto transform_opencv_pointers_Vec3d = [=](const cv::Mat& cloud, const cv::Mat& transform, cv::Mat& result) {
            const cv::Vec3d* cloudPtr = cloud.ptr<cv::Vec3d>(0);
            const double* transformPtr = transform.ptr<double>(0);
            cv::Vec3d* resultPtr = result.ptr<cv::Vec3d>(0);
            for (int i = 0; i < cloud.cols; ++i)
            {
                const cv::Vec3d& point = cloudPtr[i];

                double x = point[0];
                double y = point[1];
                double z = point[2];

                resultPtr[i][0] = transformPtr[0] * x + transformPtr[1] * y + transformPtr[2] * z + transformPtr[3];
                resultPtr[i][1] = transformPtr[4] * x + transformPtr[5] * y + transformPtr[6] * z + transformPtr[7];
                resultPtr[i][2] = transformPtr[8] * x + transformPtr[9] * y + transformPtr[10] * z + transformPtr[11];
            }
        };

        // Aplicar transformación usando Eigen
        auto transform_eigen = [=](const Eigen::MatrixXd& cloud, const Eigen::Matrix4d& transform, Eigen::MatrixXd& result) {
            result = transform * cloud;
        };

        // Aplicar transformación en paralelo
        auto transform_parallel = [=](const cv::Mat& cloud, const cv::Mat& transform, cv::Mat& result) {
            const double* cloudPtr = cloud.ptr<double>(0);
            const double* transformPtr = transform.ptr<double>(0);
            double* resultPtr = result.ptr<double>(0);
            #pragma omp parallel for
            for (int i = 0; i < cloud.cols; ++i)
            {
                double x = cloudPtr[i * 3 + 0];
                double y = cloudPtr[i * 3 + 1];
                double z = cloudPtr[i * 3 + 2];

                resultPtr[i * 3 + 0] = transformPtr[0] * x + transformPtr[1] * y + transformPtr[2] * z + transformPtr[3];
                resultPtr[i * 3 + 1] = transformPtr[4] * x + transformPtr[5] * y + transformPtr[6] * z + transformPtr[7];
                resultPtr[i * 3 + 2] = transformPtr[8] * x + transformPtr[9] * y + transformPtr[10] * z + transformPtr[11];
            }
        };

        // Aplicar transformación con CUDA
        auto transform_with_cuda = [=](const float* d_cloud, const float* d_transform, float* d_result, int num_points) {
            //...
        };

        // Kernel SIMD para transformar la nube de puntos
        auto transformWithSIMD = [=](const double* cloud, const double* transform, double* result, int num_points) {
            int num_simd_points = (num_points / 4) * 4;  // Puntos manejados con SIMD, múltiplo de 4

            // Cargar las filas de la matriz de transformación en registros SIMD
            __m256d t0 = _mm256_set1_pd(transform[0]);
            __m256d t1 = _mm256_set1_pd(transform[1]);
            __m256d t2 = _mm256_set1_pd(transform[2]);
            __m256d t3 = _mm256_set1_pd(transform[3]);

            __m256d t4 = _mm256_set1_pd(transform[4]);
            __m256d t5 = _mm256_set1_pd(transform[5]);
            __m256d t6 = _mm256_set1_pd(transform[6]);
            __m256d t7 = _mm256_set1_pd(transform[7]);

            __m256d t8 = _mm256_set1_pd(transform[8]);
            __m256d t9 = _mm256_set1_pd(transform[9]);
            __m256d t10 = _mm256_set1_pd(transform[10]);
            __m256d t11 = _mm256_set1_pd(transform[11]);

            // Procesar bloques de 4 puntos a la vez
            for (int i = 0; i < num_simd_points; i += 4) {
                // Cargar las coordenadas X, Y y Z de 4 puntos
                __m256d x = _mm256_set_pd(cloud[(i + 3) * 3], cloud[(i + 2) * 3], cloud[(i + 1) * 3], cloud[i * 3]);
                __m256d y = _mm256_set_pd(cloud[(i + 3) * 3 + 1], cloud[(i + 2) * 3 + 1], cloud[(i + 1) * 3 + 1], cloud[i * 3 + 1]);
                __m256d z = _mm256_set_pd(cloud[(i + 3) * 3 + 2], cloud[(i + 2) * 3 + 2], cloud[(i + 1) * 3 + 2], cloud[i * 3 + 2]);

                // Realizar la transformación
                __m256d res_x = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(t0, x), _mm256_mul_pd(t1, y)),
                                              _mm256_add_pd(_mm256_mul_pd(t2, z), t3));
                __m256d res_y = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(t4, x), _mm256_mul_pd(t5, y)),
                                              _mm256_add_pd(_mm256_mul_pd(t6, z), t7));
                __m256d res_z = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(t8, x), _mm256_mul_pd(t9, y)),
                                              _mm256_add_pd(_mm256_mul_pd(t10, z), t11));

                // Almacenar los resultados intercalados (X1, Y1, Z1, X2, Y2, Z2, ...)
                double temp_x[4], temp_y[4], temp_z[4];
                _mm256_store_pd(temp_x, res_x);
                _mm256_store_pd(temp_y, res_y);
                _mm256_store_pd(temp_z, res_z);

                for (int j = 0; j < 4; ++j) {
                    result[(i + j) * 3]     = temp_x[j];
                    result[(i + j) * 3 + 1] = temp_y[j];
                    result[(i + j) * 3 + 2] = temp_z[j];
                }
            }

            // Manejar los puntos restantes (si num_points no es múltiplo de 4)
            for (int i = num_simd_points; i < num_points; ++i) {
                int base = i * 3;
                double x = cloud[base];
                double y = cloud[base + 1];
                double z = cloud[base + 2];

                result[base]     = transform[0] * x + transform[1] * y + transform[2] * z + transform[3];
                result[base + 1] = transform[4] * x + transform[5] * y + transform[6] * z + transform[7];
                result[base + 2] = transform[8] * x + transform[9] * y + transform[10] * z + transform[11];
            }
        };

        // Crear nube de puntos como una sola fila
        int num_points = (int)1e6;
        cv::Mat cloud(1, num_points, CV_64FC3);
        cv::Mat transform(4, 4, CV_64F);

        // Inicializar nube de puntos y matriz de transformación
        cv::randu(cloud, cv::Scalar(0.0, 0.0, 0.0), cv::Scalar(100.0, 100.0, 100.0));

        cv::Mat cloud_homogeneous(4, cloud.cols, CV_64FC1);
        for (int i = 0; i < cloud.cols; ++i)
        {
            const cv::Vec3d& pt = cloud.at<cv::Vec3d>(0, i);
            cloud_homogeneous.at<double>(0, i) = pt[0];
            cloud_homogeneous.at<double>(1, i) = pt[1];
            cloud_homogeneous.at<double>(2, i) = pt[2];
            cloud_homogeneous.at<double>(3, i) = 1.0;
        }

        transform = cv::Mat::eye(4, 4, CV_64F);
        {
            double roll = M_PI / 6;   // Rotación alrededor del eje X
            double pitch = M_PI / 4;  // Rotación alrededor del eje Y
            double yaw = M_PI / 3;    // Rotación alrededor del eje Z
            cv::Mat Rx = (cv::Mat_<double>(3, 3) <<
                1, 0, 0,
                0, std::cos(roll), -std::sin(roll),
                0, std::sin(roll), std::cos(roll));
            cv::Mat Ry = (cv::Mat_<double>(3, 3) <<
                std::cos(pitch), 0, std::sin(pitch),
                0, 1, 0,
                -std::sin(pitch), 0, std::cos(pitch));
            cv::Mat Rz = (cv::Mat_<double>(3, 3) <<
                std::cos(yaw), -std::sin(yaw), 0,
                std::sin(yaw), std::cos(yaw), 0,
                0, 0, 1);
            cv::Mat R = Rz * Ry * Rx;
            R.copyTo(transform(cv::Rect(0, 0, 3, 3)));
            transform.at<double>(0, 3) = 10; // Traslación en x
            transform.at<double>(1, 3) = -5; // Traslación en y
            transform.at<double>(2, 3) = 3;  // Traslación en z
        }

        // Convertir a Eigen => representación matricial
        Eigen::MatrixXd cloud_eigen(4, num_points);
        Eigen::Matrix4d transformation_eigen;
        for (int i = 0; i < num_points; ++i) {
            cloud_eigen(0, i) = cloud.at<double>(0, i * 3 + 0);
            cloud_eigen(1, i) = cloud.at<double>(0, i * 3 + 1);
            cloud_eigen(2, i) = cloud.at<double>(0, i * 3 + 2);
            cloud_eigen(3, i) = 1.0;
        }

        transformation_eigen << transform.at<double>(0, 0), transform.at<double>(0, 1), transform.at<double>(0, 2), transform.at<double>(0, 3),
                                transform.at<double>(1, 0), transform.at<double>(1, 1), transform.at<double>(1, 2), transform.at<double>(1, 3),
                                transform.at<double>(2, 0), transform.at<double>(2, 1), transform.at<double>(2, 2), transform.at<double>(2, 3),
                                transform.at<double>(3, 0), transform.at<double>(3, 1), transform.at<double>(3, 2), transform.at<double>(3, 3);

        /// Pruebas antiguas
        {
//            // OpenCV directo
//            cv::Mat result_opencv;
//            result_opencv.create(cloud_homogeneous.rows, cloud_homogeneous.cols, CV_64FC1);
//            double time_opencv = measure_time([&]() {
//                transform_opencv(cloud_homogeneous, transform, result_opencv);
//            });

//            // OpenCV puntero double
//            cv::Mat result_opencv_pt_d;
//            result_opencv_pt_d.create(cloud.rows, cloud.cols, CV_64FC3);
//            double time_opencv_pt_d = measure_time([&]() {
//                transform_opencv_pointers_double(cloud, transform, result_opencv_pt_d);
//            });

//            // OpenCV puntero Vec3d
//            cv::Mat result_opencv_pt_V;
//            result_opencv_pt_V.create(cloud.rows, cloud.cols, CV_64FC3);
//            double time_opencv_pt_V = measure_time([&]() {
//                transform_opencv_pointers_Vec3d(cloud, transform, result_opencv_pt_V);
//            });

//            // Eigen
//            Eigen::MatrixXd result_eigen(4, num_points);
//            double time_eigen = measure_time([&]() {
//                transform_eigen(cloud_eigen, transformation_eigen, result_eigen);
//            });

//            // Paralelo
//            cv::Mat result_parallel;
//            result_parallel.create(cloud.rows, cloud.cols, CV_64FC3);
//            double time_parallel = measure_time([&]() {
//                transform_parallel(cloud, transform, result_parallel);
//            });

//            // SIMD
//            cv::Mat result_SIMD;
//            result_SIMD.create(cloud.rows, cloud.cols, CV_64FC3);

//            double time_SIMD = measure_time([&]() {
//                transformWithSIMD(cloud.ptr<double>(), transform.ptr<double>(), result_SIMD.ptr<double>(), num_points);
//            });

//            // Imprimir por rapidez descendente
//            std::vector<std::pair<std::string, double>> times = {
//                {"OpenCV directo", time_opencv},
//                {"OpenCV puntero double", time_opencv_pt_d},
//                {"OpenCV puntero Vec3d", time_opencv_pt_V},
//                {"Eigen", time_eigen},
//                {"Paralelo", time_parallel},
//                {"SIMD", time_SIMD}
//            };
//            std::sort(times.begin(), times.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
//                return a.second < b.second; // Ordenar de mayor a menor
//            });
//            std::cout << "\n";
//            for (const auto& pair : times)
//                std::cout << "Tiempo " << pair.first << ": " << pair.second << " segundos" << std::endl;

//            auto compare_points = [=](const cv::Vec3d& p1, const cv::Vec3d& p2, double tolerance = 1e-3) {
//                return std::fabs(p1[0] - p2[0]) < tolerance &&
//                       std::fabs(p1[1] - p2[1]) < tolerance &&
//                       std::fabs(p1[2] - p2[2]) < tolerance;
//            };
//            std::cout << "\nSe imprimirá error si hay puntos diferentes..." << std::endl;
//            for (int i = 0; i < num_points; ++i)
//            {
//                cv::Vec3d point_opencv = cv::Vec3d(result_opencv.at<double>(0, i), result_opencv.at<double>(1, i), result_opencv.at<double>(2, i));
//                cv::Vec3d point_opencv_pt_d = result_opencv_pt_d.at<cv::Vec3d>(0, i);
//                cv::Vec3d point_opencv_pt_V = result_opencv_pt_V.at<cv::Vec3d>(0, i);
//                Eigen::Vector4d point_eigen = result_eigen.col(i);
//                cv::Vec3d point_parallel = result_parallel.at<cv::Vec3d>(0, i);
//                cv::Vec3d point_SIMD = result_SIMD.at<cv::Vec3d>(0, i);

//                if (!compare_points(point_opencv, point_opencv_pt_d) || !compare_points(point_opencv, point_opencv_pt_V) ||
//                    !compare_points(point_opencv, cv::Vec3d(point_eigen(0), point_eigen(1), point_eigen(2))) ||
//                    !compare_points(point_opencv, point_parallel) || !compare_points(point_opencv, point_SIMD))
//                {
//                    std::cerr << "Error en el punto " << i << " entre los diferentes métodos." << std::endl;
//                    std::cerr << "OpenCV directo: (" << result_opencv.at<double>(0, i) << ", " << result_opencv.at<double>(1, i) << ", " << result_opencv.at<double>(2, i) << ")" << std::endl;
//                    std::cerr << "OpenCV puntero double: (" << point_opencv_pt_d[0] << ", " << point_opencv_pt_d[1] << ", " << point_opencv_pt_d[2] << ")" << std::endl;
//                    std::cerr << "OpenCV puntero Vec3d: (" << point_opencv_pt_V[0] << ", " << point_opencv_pt_V[1] << ", " << point_opencv_pt_V[2] << ")" << std::endl;
//                    std::cerr << "Eigen: (" << point_eigen(0) << ", " << point_eigen(1) << ", " << point_eigen(2) << ")" << std::endl;
//                    std::cerr << "Paralelo: (" << point_parallel[0] << ", " << point_parallel[1] << ", " << point_parallel[2] << ")" << std::endl;
//                    std::cerr << "SIMD: (" << point_SIMD[0] << ", " << point_SIMD[1] << ", " << point_SIMD[2] << ")" << std::endl;
//                }
//            }
//            int pointToPick{50};
//            cv::Vec3d point_opencv = cv::Vec3d(result_opencv.at<double>(0, pointToPick), result_opencv.at<double>(1, pointToPick), result_opencv.at<double>(2, pointToPick));
//            cv::Vec3d point_opencv_pt_d = result_opencv_pt_d.at<cv::Vec3d>(0, pointToPick);
//            cv::Vec3d point_opencv_pt_V = result_opencv_pt_V.at<cv::Vec3d>(0, pointToPick);
//            Eigen::Vector4d point_eigen = result_eigen.col(pointToPick);
//            cv::Vec3d point_parallel = result_parallel.at<cv::Vec3d>(0, pointToPick);
//            cv::Vec3d point_SIMD = result_SIMD.at<cv::Vec3d>(0, pointToPick);
//            std::cout << "\nPunto " << pointToPick << " entre los diferentes métodos." << std::endl;
//            std::cout << "OpenCV directo: (" << result_opencv.at<double>(0, pointToPick) << ", " << result_opencv.at<double>(1, pointToPick) << ", " << result_opencv.at<double>(2, pointToPick) << ")" << std::endl;
//            std::cout << "OpenCV puntero double: (" << point_opencv_pt_d[0] << ", " << point_opencv_pt_d[1] << ", " << point_opencv_pt_d[2] << ")" << std::endl;
//            std::cout << "OpenCV puntero Vec3d: (" << point_opencv_pt_V[0] << ", " << point_opencv_pt_V[1] << ", " << point_opencv_pt_V[2] << ")" << std::endl;
//            std::cout << "Eigen: (" << point_eigen(0) << ", " << point_eigen(1) << ", " << point_eigen(2) << ")" << std::endl;
//            std::cout << "Paralelo: (" << point_parallel[0] << ", " << point_parallel[1] << ", " << point_parallel[2] << ")" << std::endl;
//            std::cout << "SIMD: (" << point_SIMD[0] << ", " << point_SIMD[1] << ", " << point_SIMD[2] << ")" << std::endl;
        }

        /// Pruebas nuevas
        {
            const int num_iters = 10000;
            std::vector<int> rankingOpenCVDirect, rankingOpenCVPtr, rankingOpenCVPtrVec, rankingEigen, rankingParallel, rankingSIMD;
            double avgOpenCVDirect = 0.0, avgOpenCVPtr = 0.0, avgOOpenCVPtrVec = 0.0, avgEigen = 0.0, avgParallel = 0.0, avgSIMD = 0.0;

            std::ofstream outputFile("/home/alejandro/Escritorio/fast3d/benchmark_results.txt");
            if (!outputFile.is_open()) {
                std::cerr << "Error al abrir benchmark_results.\n";
                return 1;
            }

            std::ofstream averagesOutputFile("/home/alejandro/Escritorio/fast3d/averages_results.txt");
            if (!outputFile.is_open()) {
                std::cerr << "Error al abrir averages_results.\n";
                return 1;
            }

            for (int i = 0; i < num_iters; ++i)
            {
                // OpenCV directo
                cv::Mat result_opencv;
                result_opencv.create(cloud_homogeneous.rows, cloud_homogeneous.cols, CV_64FC1);
                double time_opencv = measure_time([&]() {
                    transform_opencv(cloud_homogeneous, transform, result_opencv);
                });

                // OpenCV puntero double
                cv::Mat result_opencv_pt_d;
                result_opencv_pt_d.create(cloud.rows, cloud.cols, CV_64FC3);
                double time_opencv_pt_d = measure_time([&]() {
                    transform_opencv_pointers_double(cloud, transform, result_opencv_pt_d);
                });

                // OpenCV puntero Vec3d
                cv::Mat result_opencv_pt_V;
                result_opencv_pt_V.create(cloud.rows, cloud.cols, CV_64FC3);
                double time_opencv_pt_V = measure_time([&]() {
                    transform_opencv_pointers_Vec3d(cloud, transform, result_opencv_pt_V);
                });

                // Eigen
                Eigen::MatrixXd result_eigen(4, num_points);
                double time_eigen = measure_time([&]() {
                    transform_eigen(cloud_eigen, transformation_eigen, result_eigen);
                });

                // Paralelo
                cv::Mat result_parallel;
                result_parallel.create(cloud.rows, cloud.cols, CV_64FC3);
                double time_parallel = measure_time([&]() {
                    transform_parallel(cloud, transform, result_parallel);
                });

                // SIMD
                cv::Mat result_SIMD;
                result_SIMD.create(cloud.rows, cloud.cols, CV_64FC3);

                double time_SIMD = measure_time([&]() {
                    transformWithSIMD(cloud.ptr<double>(), transform.ptr<double>(), result_SIMD.ptr<double>(), num_points);
                });

                avgOpenCVDirect += time_opencv;
                avgOpenCVPtr += time_opencv_pt_d;
                avgOOpenCVPtrVec += time_opencv_pt_V;
                avgEigen += time_eigen;
                avgParallel += time_parallel;
                avgSIMD += time_SIMD;

                // Ranking por tiempo (menor es más rápido)
                std::vector<std::pair<std::string, double>> timeResults = {
                    {"OpenCV directo", time_opencv},
                    {"OpenCV puntero double", time_opencv_pt_d},
                    {"OpenCV puntero Vec3d", time_opencv_pt_V},
                    {"Eigen", time_eigen},
                    {"Paralelo", time_parallel},
                    {"SIMD", time_SIMD}
                };
                std::sort(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                    return a.second < b.second; // Ordenar de mayor a menor
                });

                // Asignar un ranking (basado en la posición) a cada técnica de un conjunto de 6, buscando el nombre de cada técnica en el vector timeResults y luego calculando su posición (ranking) en función de su orden en el tiempo
                rankingOpenCVDirect.push_back(std::distance(timeResults.begin(), std::find_if(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& p) { return p.first == "OpenCV directo"; })) + 1);
                rankingOpenCVPtr.push_back(std::distance(timeResults.begin(), std::find_if(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& p) { return p.first == "OpenCV puntero double"; })) + 1);
                rankingOpenCVPtrVec.push_back(std::distance(timeResults.begin(), std::find_if(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& p) { return p.first == "OpenCV puntero Vec3d"; })) + 1);
                rankingEigen.push_back(std::distance(timeResults.begin(), std::find_if(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& p) { return p.first == "Eigen"; })) + 1);
                rankingParallel.push_back(std::distance(timeResults.begin(), std::find_if(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& p) { return p.first == "Paralelo"; })) + 1);
                rankingSIMD.push_back(std::distance(timeResults.begin(), std::find_if(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& p) { return p.first == "SIMD"; })) + 1);
            }

            avgOpenCVDirect /= num_iters;
            avgOpenCVPtr /= num_iters;
            avgOOpenCVPtrVec /= num_iters;
            avgEigen /= num_iters;
            avgParallel /= num_iters;
            avgSIMD /= num_iters;

            // Guardar los promedios en un archivo
            averagesOutputFile << "avgOpenCVDirect, avgOpenCVPtr, "
                << "avgOOpenCVPtrVec, avgEigen, "
                << "avgParallel, avgSIMD\n";
            averagesOutputFile << std::fixed << std::setprecision(6);
            averagesOutputFile << avgOpenCVDirect << ", "
                       << avgOpenCVPtr << ", "
                       << avgOOpenCVPtrVec << ", "
                       << avgEigen << ", "
                       << avgParallel << ", "
                       << avgSIMD << "\n";
            averagesOutputFile.close();

            // Guardar los rankings en un archivo
            for (int i = 0; i < num_iters; ++i) {
                outputFile << i + 1 << ", "
                           << rankingOpenCVDirect[i] << ", "
                           << rankingOpenCVPtr[i] << ", "
                           << rankingOpenCVPtrVec[i] << ", "
                           << rankingEigen[i] << ", "
                           << rankingParallel[i] << ", "
                           << rankingSIMD[i] << "\n";
            }
            outputFile.close();
        }
    }

    return 0;
}

