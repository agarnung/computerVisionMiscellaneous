#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <eigen3/Eigen/Dense>

std::vector<std::vector<double>> generarToroide(double c, double a, int numPuntos, std::vector<double> centro = {0.0, 0.0, 0.0}) {
    std::vector<std::vector<double>> puntos;
    std::srand(static_cast<unsigned>(std::time(0)));

    for (int i = 0; i < numPuntos; ++i) {
        double theta = (static_cast<double>(std::rand()) / RAND_MAX) * 2.0 * M_PI;
        double phi = (static_cast<double>(std::rand()) / RAND_MAX) * 2.0 * M_PI;

        double x = (c + a * std::cos(phi)) * std::cos(theta) + centro[0];
        double y = (c + a * std::cos(phi)) * std::sin(theta) + centro[1];
        double z = a * std::sin(phi) + centro[2];

        puntos.push_back({x, y, z});
    }

    return puntos;
}

std::vector<std::vector<double>> añadirRuidoGaussiano(const std::vector<std::vector<double>>& puntos, double desviacion) {
    std::vector<std::vector<double>> puntosRuidosos;
    std::srand(static_cast<unsigned>(std::time(0)));

    // Media 0
    for (const auto& punto : puntos) {
        double ruidoX = (static_cast<double>(std::rand()) / RAND_MAX) * desviacion * 2.0 - desviacion;
        double ruidoY = (static_cast<double>(std::rand()) / RAND_MAX) * desviacion * 2.0 - desviacion;
        double ruidoZ = (static_cast<double>(std::rand()) / RAND_MAX) * desviacion * 2.0 - desviacion;

        puntosRuidosos.push_back({punto[0] + ruidoX, punto[1] + ruidoY, punto[2] + ruidoZ});
    }

    return puntosRuidosos;
}

std::vector<std::vector<double>> eliminarPuntosAleatorios(const std::vector<std::vector<double>>& puntos, double porcentajeEliminacion) {
    std::vector<std::vector<double>> puntosRestantes;
    std::srand(static_cast<unsigned>(std::time(0)));

    int num_puntos_a_eliminar = static_cast<int>(puntos.size() * porcentajeEliminacion);

    std::vector<std::vector<double>> puntosCopia = puntos;

    for (int i = 0; i < num_puntos_a_eliminar; ++i) {
        int indiceAleatorio = std::rand() % puntosCopia.size();
        puntosCopia.erase(puntosCopia.begin() + indiceAleatorio);
    }

    return puntosCopia;
}

std::vector<std::vector<double>> trasladarPuntos(const std::vector<std::vector<double>>& puntos, double tx, double ty, double tz) {
    std::vector<std::vector<double>> puntosTraslados;

    for (const auto& punto : puntos) {
        std::vector<double> puntoTrasladado = {
            punto[0] + tx,
            punto[1] + ty,
            punto[2] + tz
        };

        puntosTraslados.push_back(puntoTrasladado);
    }

    return puntosTraslados;
}

void guardarNubePuntos(const std::vector<std::vector<double>>& puntos, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo para escribir." << std::endl;
        return;
    }

    for (const auto& punto : puntos) {
        file << punto[0] << " " << punto[1] << " " << punto[2] << "\n";
    }

    file.close();
    std::cout << "Nube de puntos guardada en: " << filename << std::endl;
}

std::vector<std::vector<double>> leerNubePuntos(const std::string& filename) {
    std::vector<std::vector<double>> puntos;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir el archivo para leer." << std::endl;
        return puntos;
    }

    double x, y, z;
    while (file >> x >> y >> z) {
        puntos.push_back({x, y, z});
    }

    file.close();
    std::cout << "Nube de puntos leída desde: " << filename << std::endl;
    return puntos;
}

bool fitTorus(const std::vector<std::vector<double>>& puntos, double& a, double& c) {
    int num_points = puntos.size();
    if (num_points < 3) return false;

    Eigen::MatrixXd points(num_points, 3);
    for (int i = 0; i < num_points; ++i) {
        points(i, 0) = puntos[i][0];
        points(i, 1) = puntos[i][1];
        points(i, 2) = puntos[i][2];
    }

    Eigen::MatrixXd A(num_points, 3);
    Eigen::VectorXd b(num_points);

    for (int i = 0; i < num_points; ++i) {
        double xi = points(i, 0);
        double yi = points(i, 1);
        double zi = points(i, 2);

        double rxy = std::sqrt(xi * xi + yi * yi);

        A(i, 0) = 1.0;
        A(i, 1) = -2.0 * rxy;
        A(i, 2) = -1.0;

        b(i) = -(xi * xi + yi * yi + zi * zi);
    }

    if (A.rows() == 0 || A.cols() == 0 || b.size() != A.rows()) return false;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.rank() < 2) return false;

    Eigen::VectorXd params = svd.solve(b);

    a = std::sqrt(params(0));
    c = params(1);

    if (!(std::isfinite(a) && std::isfinite(c))) return false;

    return true;
}

bool fitTorusUncentered(const std::vector<std::vector<double>>& puntos, double& a, double& c, double& xo, double& yo, double& zo) {
    int num_points = puntos.size();
    if (num_points < 8) {
        return false;
    }

    Eigen::MatrixXd points(num_points, 3);
    for (int j = 0; j < num_points; ++j) {
        points(j, 0) = puntos[j][0];
        points(j, 1) = puntos[j][1];
        points(j, 2) = puntos[j][2];
    }

    Eigen::Vector3d centroid = points.colwise().mean();
    const double xo_0 = centroid(0);
    const double yo_0 = centroid(1);

    const double xo_0_squared_plus_yo_0_squared = xo_0 * xo_0 + yo_0 * yo_0;

    Eigen::MatrixXd A(num_points, 8);
    Eigen::VectorXd b(num_points);
    for (int i = 0; i < num_points; ++i) {
        const double xi = points(i, 0);
        const double yi = points(i, 1);
        const double zi = points(i, 2);

        A(i, 0) = 1.0;
        A(i, 1) = 1.0;
        A(i, 2) = -2.0 * xi;
        A(i, 3) = -2.0 * yi;
        A(i, 4) = -2.0 * zi;

        const double Chi = xo_0_squared_plus_yo_0_squared - xi * xo_0 - yi * yo_0;
        const double Li = std::sqrt((xi - xo_0) * (xi - xo_0) + (yi - yo_0) * (yi - yo_0));
        A(i, 5) = 2.0 * (Chi - Li * Li) / Li;

        A(i, 6) = 2.0 * (xi - xo_0) / Li;
        A(i, 7) = 2.0 * (yi - yo_0) / Li;

        b(i) = xi * xi + yi * yi + zi * zi;
    }

    if (A.rows() == 0 || A.cols() == 0 || b.size() != A.rows()) {
        return false;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.rank() < 4) {
        return false;
    }

    Eigen::VectorXd params = svd.solve(b);

    xo = -1 * params(2);
    yo = -1 * params(3);
    zo = -1 * params(4);
    c = std::abs(params(6) / xo);
    a = std::sqrt(c * c / std::abs(params(0)));

    if (!(std::isfinite(a) && std::isfinite(c) && std::isfinite(xo) && std::isfinite(yo) && std::isfinite(zo))) {
        return false;
    }

    return true;
}

int main() {
    std::cout << "Ecuación del toroide: (c - sqrt((x - xo)^2 + (y - yo)^2))^2 + (z - zo)^2 = a^2, r = c - a, R = c + a\n";

    double c = 5.0;
    double a = 2.0;
    int numPuntos = 10000;
    double std_dev = 0.1;
    double porcentajeEliminacion = 0.1;

    std::vector<std::vector<double>> toroide = generarToroide(c, a, numPuntos);
    std::vector<std::vector<double>> toroideRuidoso = añadirRuidoGaussiano(toroide, std_dev);
    toroideRuidoso = eliminarPuntosAleatorios(toroideRuidoso, porcentajeEliminacion);

    std::string archivoIdeal = "/opt/proyectos/computerVisionMiscellaneous/2025-26-01-torus-fit/data/ideal_torus.txt";
    std::string archivoRuidoso = "/opt/proyectos/computerVisionMiscellaneous/2025-26-01-torus-fit/data/ruidoso_torus.txt";
    guardarNubePuntos(toroide, archivoIdeal);
    guardarNubePuntos(toroideRuidoso, archivoRuidoso);

    double fitted_a, fitted_c;
    if (fitTorus(toroideRuidoso, fitted_a, fitted_c)) {
        std::cout << "Ajuste de toroide exitoso:\n";
        std::cout << "a = " << fitted_a << ", c = " << fitted_c << std::endl;

        std::vector<std::vector<double>> toroideAjustado = generarToroide(fitted_c, fitted_a, numPuntos);

        std::string archivoAjustado = "/opt/proyectos/computerVisionMiscellaneous/2025-26-01-torus-fit/data/ajustado_torus.txt";
        guardarNubePuntos(toroideAjustado, archivoAjustado);
    } else {
        std::cout << "Fallo en el ajuste del toroide." << std::endl;
    }

    std::string archivoRuidosoTrasladado = "/opt/proyectos/computerVisionMiscellaneous/2025-26-01-torus-fit/data/ruidosoTrasladado_torus.txt";
    std::vector<std::vector<double>> toroideRuidosoTrasladado = trasladarPuntos(toroideRuidoso, 10.0, -2.0, 3.0);
    guardarNubePuntos(toroideRuidosoTrasladado, archivoRuidosoTrasladado);

    double xo, yo, zo;
    if (fitTorusUncentered(toroideRuidosoTrasladado, fitted_a, fitted_c, xo, yo, zo)) {
        std::cout << "Ajuste de toroide descentrado exitoso:\n";
        std::cout << "a = " << fitted_a << ", c = " << fitted_c << ", xo = " << xo << ", yo = " << yo << ", zo = " << zo << std::endl;

        std::vector<std::vector<double>> toroideAjustado = generarToroide(fitted_c, fitted_a, numPuntos, std::vector<double>{xo, yo, zo});

        std::string archivoAjustado = "/opt/proyectos/computerVisionMiscellaneous/2025-26-01-torus-fit/data/ajustado_torus_uncentered.txt";
        guardarNubePuntos(toroideAjustado, archivoAjustado);
    } else {
        std::cout << "Fallo en el ajuste del toroide descentrado." << std::endl;
    }

    /// Caso de uso donde funcionaría bien: zona más o menos plana:
    std::string regionPlanaPath = "/opt/proyectos/computerVisionMiscellaneous/2025-26-01-torus-fit/rp2.asc";
    std::vector<std::vector<double>> regionPlana = leerNubePuntos(regionPlanaPath);

    // if (fitTorus(regionPlana, fitted_a, fitted_c)) {
    if (fitTorusUncentered(regionPlana, fitted_a, fitted_c, xo, yo, zo)) {
        std::cout << "Ajuste de toroide de región plana exitoso:\n";
        std::cout << "a = " << fitted_a << ", c = " << fitted_c << std::endl;

        std::vector<std::vector<double>> toroideAjustado = generarToroide(fitted_c, fitted_a, numPuntos);

        std::string archivoAjustado = "/opt/proyectos/computerVisionMiscellaneous/2025-26-01-torus-fit/data/ajustado_region_plana_torus.txt";
        guardarNubePuntos(toroideAjustado, archivoAjustado);
    } else {
        std::cout << "Fallo en el ajuste del toroide de región plana." << std::endl;
    }

    return 0;
}
