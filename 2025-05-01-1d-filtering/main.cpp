//- Proyecto de métodos filtrado 1D, en c++ y python, meter:
// - https://github.com/t-suzuki/total_variation_test/blob/master/total_variation.py
// - https://github.com/BioWar/l0_gradient_minimization_test
// - Mis filtrados de REM
// - Lo más básico (media, media movil, mediana, moda, min/max...)
// - Todos con test unitarios y datos de prueba y que haya una función para generar señales aleatorias

#include <iostream>
#include <random>
#include <vector>
#include <opencv2/opencv.hpp>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

void denoiseTV(std::vector<double>& u, const std::vector<double>& f, double lambda, double step_size, int iterations)
{
    size_t N = u.size();

    auto sgn = [](double x) {
        if (x > 0) return 1;
        if (x < 0) return -1;
        return 0;
    };

    auto TV_derivative = [&](const std::vector<double>& u, size_t i) {
        if (i == 0) {
            return sgn(u[i] - u[i+1]); // Primer elemento, diferencia con el siguiente
        }
        if (i == u.size() - 1) {
            return sgn(u[i] - u[i-1]); // Último elemento, diferencia con el anterior
        }
        return sgn(u[i] - u[i-1]) - sgn(u[i+1] - u[i]); // Elementos intermedios
    };

    auto calculateEnergy = [&](const std::vector<double>& u, const std::vector<double>& f) {
        double tv_energy = 0.0;
        double fidelity_energy = 0.0;

        for (size_t i = 1; i < u.size(); ++i) {
            tv_energy += std::abs(u[i] - u[i-1]);
            fidelity_energy += lambda * 0.5 * std::pow(u[i] - f[i], 2);
        }

        return tv_energy + fidelity_energy;
    };

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<double> u_new(N);

        for (size_t i = 1; i < N - 1; ++i) {
            double tv_grad = TV_derivative(u, i);  // Calcular el gradiente de la TV
            u_new[i] = u[i] - step_size * (tv_grad + lambda * (u[i] - f[i]));  // Actualizar la señal
        }

        u_new[0] = u[0] - step_size * (lambda * (u[0] - f[0]));  // Frontera izquierda
        u_new[N-1] = u[N-1] - step_size * (lambda * (u[N-1] - f[N-1]));  // Frontera derecha

        u = u_new;
    }
}

void denoiseTV_EL(std::vector<double>& u, const std::vector<double>& f, double lambda, double step_size, int iterations)
{
    size_t N = u.size();

    auto grad_u_Neumann = [&](const std::vector<double>& u, size_t i) {
        if (i < 1 || i > N - 1)
            return 0.0;
        else
            return u[i] - u[i - 1];
    };

    auto grad_TV = [&](const std::vector<double>& u, size_t i) {
        double grad = grad_u_Neumann(u, i);
        double mag = std::abs(grad);
        return mag == 0 ? 0.0 : grad / mag;
    };

    auto calculateEnergy = [&](const std::vector<double>& u, const std::vector<double>& f) {
        double tv_energy = 0.0;
        double fidelity_energy = 0.0;

        for (size_t i = 1; i < u.size(); ++i)
        {
            tv_energy += std::abs(u[i] - u[i-1]);
            fidelity_energy += lambda * 0.5 * std::pow(u[i] - f[i], 2);
        }

        return tv_energy + fidelity_energy;
    };
    (void)calculateEnergy;

    for (int iter = 0; iter < iterations; ++iter)
    {
        std::vector<double> u_new(N);

        for (size_t i = 1; i < N - 1; ++i)
        {
            double tv_grad = grad_TV(u, i);
            u_new[i] = u[i] - step_size * (tv_grad + lambda * (u[i] - f[i]));
        }

        u_new[0] = u[0] - step_size * (lambda * (u[0] - f[0]));
        u_new[N - 1] = u[N - 1] - step_size * (lambda * (u[N - 1] - f[N - 1]));

        u = u_new;
    }
}

void convexHull1D(const std::vector<double>& noisy_signal, std::vector<double>& convex_signal)
{
    // Convertir la señal 1D en un conjunto de puntos 2D.
    // Cada punto tiene coordenadas (x, y), donde x es el índice y y es el valor de la señal.
    std::vector<cv::Point2f> points;
    for (int i = 0; i < (int)noisy_signal.size(); ++i)
        points.emplace_back(static_cast<float>(i), static_cast<float>(noisy_signal[i]));

    // Calcular el convex hull (envolvente convexa) de los puntos 2D.
    // El convex hull es un polígono convexo que envuelve todos los puntos.
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);

    // Limpiar y redimensionar el vector de salida para almacenar la señal convexa.
    convex_signal.clear();
    convex_signal.resize(noisy_signal.size());

    // Para cada punto en la señal original, calcular su valor en la envolvente convexa.
    for (int i = 0; i < (int)noisy_signal.size(); ++i)
    {
        // Convertir el índice actual a un valor flotante (coordenada x).
        const float x = static_cast<float>(i);

        // Inicializar el valor máximo de y con un valor muy pequeño.
        float max_y = -std::numeric_limits<float>::infinity();

        // Recorrer los bordes del convex hull para encontrar el valor máximo de y en x.
        for (size_t j = 0; j < hull.size(); ++j)
        {
            // Obtener el índice del siguiente vértice en el convex hull (cíclico).
            const size_t next_j = (j + 1) % hull.size();

            // Obtener los dos vértices que forman el borde actual.
            const cv::Point2f& p1 = hull[j];
            const cv::Point2f& p2 = hull[next_j];

            // Verificar si x está dentro del rango de x de este borde.
            // Si x está dentro del rango de una arista, calculamos el valor de "y"
            // correspondiente a ese x usando interpolación lineal entre los dos
            // vértices de la arista.
            if ((p1.x <= x && x <= p2.x) || (p2.x <= x && x <= p1.x))
            {
                const float t = (x - p1.x) / (p2.x - p1.x);

                // Calcular el valor de y interpolado en el borde actual.
                const float y = p1.y + t * (p2.y - p1.y);

                // Actualizar el valor máximo de "y" si el valor interpolado es mayor.
                if (y > max_y)
                    max_y = y;
            }
        }

        // Almacenar el valor máximo de "y" en la señal convexa.
        convex_signal[i] = static_cast<double>(max_y);
    }
}

void applyMovingAverageFilter(const std::vector<double>& signal, std::vector<double>& smoothed_signal, int window_size)
{
    smoothed_signal = std::vector<double>(signal.size(), 0.0);
    int half_window = window_size / 2;

    for (size_t i = 0; i < signal.size(); ++i)
    {
        double sum = 0.0;
        int count = 0;
        for (int j = -half_window; j <= half_window; ++j)
        {
            int idx = i + j;
            if (idx >= 0 && idx < (int)signal.size())
            {
                sum += signal[idx];
                count++;
            }
        }

        smoothed_signal[i] = sum / count;
    }
}

void applyMovingMaxFilter(const std::vector<double>& signal, std::vector<double>& filtered_signal, int window_size)
{
    filtered_signal = std::vector<double>(signal.size(), 0.0);
    int half_window = window_size / 2;

    for (size_t i = 0; i < signal.size(); ++i)
    {
        double max_val = -std::numeric_limits<double>::infinity();
        for (int j = -half_window; j <= half_window; ++j)
        {
            int idx = i + j;
            if (idx >= 0 && idx < (int)signal.size())
                max_val = std::max(max_val, signal[idx]);
        }

        filtered_signal[i] = max_val;
    }
}

void applyMovingMinFilter(const std::vector<double>& signal, std::vector<double>& filtered_signal, int window_size)
{
    filtered_signal = std::vector<double>(signal.size(), 0.0);
    int half_window = window_size / 2;

    for (size_t i = 0; i < signal.size(); ++i)
    {
        double min_val = std::numeric_limits<double>::infinity();
        for (int j = -half_window; j <= half_window; ++j)
        {
            int idx = i + j;
            if (idx >= 0 && idx < (int)signal.size())
                min_val = std::min(min_val, signal[idx]);
        }

        filtered_signal[i] = min_val;
    }
}

// L0 gradient minimization
// @see https://github.com/BioWar/l0_gradient_minimization_test/tree/master

// ------------------------- Funciones 1D -------------------------

// Desplazamiento circulante: rota el vector de forma circular.
// Se utiliza la aritmética modular para manejar desplazamientos negativos.
std::vector<double> circulantShift(const std::vector<double>& xs, int h) {
    int n = xs.size();
    int shift = ((h % n) + n) % n; // asegura un desplazamiento positivo
    std::vector<double> result;
    result.insert(result.end(), xs.begin() + shift, xs.end());
    result.insert(result.end(), xs.begin(), xs.begin() + shift);
    return result;
}

// Calcula la diferencia entre el vector desplazado y el original.
std::vector<double> circulant_dx(const std::vector<double>& xs, int h) {
    std::vector<double> shifted = circulantShift(xs, h);
    int n = xs.size();
    std::vector<double> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = shifted[i] - xs[i];
    }
    return result;
}

// Funciones de FFT usando OpenCV para datos 1D

// Realiza la FFT de un vector 1D (se representa como un cv::Mat de 1xN de tipo CV_64F).
// El resultado es un cv::Mat 1xN de tipo CV_64FC2 (complejo).
cv::Mat fft1D(const std::vector<double>& x) {
    cv::Mat xMat(1, static_cast<int>(x.size()), CV_64F);
    for (int i = 0; i < (int)x.size(); ++i) {
        xMat.at<double>(0, i) = x[i];
    }
    cv::Mat X;
    cv::dft(xMat, X, cv::DFT_COMPLEX_OUTPUT);
    return X;
}

// Realiza la transformada inversa (IFFT) y devuelve únicamente la parte real.
std::vector<double> ifft1D(const cv::Mat& X) {
    cv::Mat xMat;
    cv::dft(X, xMat, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    std::vector<double> x(xMat.cols);
    for (int i = 0; i < xMat.cols; ++i) {
        x[i] = xMat.at<double>(0, i);
    }
    return x;
}

// psf2otf: convierte un PSF (vector de tamaño menor) a un OTF de tamaño N.
// Primero se realiza un padding con ceros y luego se hace un desplazamiento circular
// para centrar el kernel antes de aplicar la FFT.
cv::Mat psf2otf(const std::vector<double>& psf, int N) {
    int n = psf.size();
    // Crear vector con padding de longitud N
    std::vector<double> pad(N, 0.0);
    for (int i = 0; i < n && i < N; ++i) {
        pad[i] = psf[i];
    }
    // Desplazamiento circular: concatenar pad[n/2:] y pad[:n/2]
    int shiftIndex = n / 2;
    std::vector<double> shifted;
    // Desde shiftIndex hasta el final (se consideran también los ceros si N > n)
    shifted.insert(shifted.end(), pad.begin() + shiftIndex, pad.end());
    // Luego los primeros shiftIndex elementos
    shifted.insert(shifted.end(), pad.begin(), pad.begin() + shiftIndex);

    // Copiar a un cv::Mat de 1xN
    cv::Mat otfInput(1, N, CV_64F);
    for (int i = 0; i < N; ++i) {
        otfInput.at<double>(0, i) = shifted[i];
    }

    // Calcular la FFT para obtener el OTF
    cv::Mat otf;
    cv::dft(otfInput, otf, cv::DFT_COMPLEX_OUTPUT);
    return otf;
}

// ------------------------- Minimización de Gradiente L₀ en 1D -------------------------

// Implementa el algoritmo de minimización de gradiente L₀ en 1D.
// I: vector de entrada
// lmd: parámetro lambda
// beta_max: valor máximo de beta
// beta_rate: tasa de incremento de beta en cada iteración (por defecto 2.0)
// max_iter: número máximo de iteraciones (por defecto 30)
// return_history: si se desea retornar la historia de S en cada iteración (aquí se retorna solo S final)
std::vector<double> l0_gradient_minimization_1d(const std::vector<double>& I, double lmd, double beta_max, double beta_rate = 2.0, int max_iter = 30, bool return_history = false) {
    // S es la copia de I
    std::vector<double> S = I;
    int N = S.size();

    // Preparar la FFT de la imagen de entrada
    cv::Mat F_I = fft1D(S);

    // Calcular F_denom = |psf2otf([-1, 1], N)|^2
    std::vector<double> psf = {-1.0, 1.0};
    cv::Mat otf = psf2otf(psf, N); // Resultado complejo 1xN
    std::vector<double> F_denom(N, 0.0);
    for (int i = 0; i < N; ++i) {
        cv::Vec2d val = otf.at<cv::Vec2d>(0, i);
        F_denom[i] = val[0] * val[0] + val[1] * val[1];
    }

    double beta = lmd * 2.0;
    // Para almacenar la historia de S (si se requiere)
    std::vector< std::vector<double> > S_history;
    if (return_history) {
        S_history.push_back(S);
    }

    std::vector<double> hp(N, 0.0);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Con S, se resuelve para hp: hp = circulant_dx(S, 1)
        hp = circulant_dx(S, 1);
        // Aplicar umbral: si hp[i]^2 < lmd/beta, entonces hp[i] = 0
        for (int i = 0; i < N; ++i) {
            if (hp[i] * hp[i] < lmd / beta) {
                hp[i] = 0.0;
            }
        }

        // Calcular FFT de circulant_dx(hp, -1)
        std::vector<double> shifted_hp = circulant_dx(hp, -1);
        cv::Mat fft_hp = fft1D(shifted_hp);

        // Calcular numerador: F_I + beta * fft_hp
        cv::Mat numerator = F_I + beta * fft_hp;

        // Dividir elemento a elemento por: 1.0 + beta * F_denom
        cv::Mat numerator_divided = numerator.clone(); // mismo tamaño y tipo que numerator
        for (int i = 0; i < N; ++i) {
            double denom = 1.0 + beta * F_denom[i];
            cv::Vec2d num = numerator.at<cv::Vec2d>(0, i);
            numerator_divided.at<cv::Vec2d>(0, i)[0] = num[0] / denom;
            numerator_divided.at<cv::Vec2d>(0, i)[1] = num[1] / denom;
        }

        // Actualizar S usando la IFFT (tomando la parte real)
        S = ifft1D(numerator_divided);

        if (return_history) {
            S_history.push_back(S);
        }

        beta *= beta_rate;
        if (beta > beta_max) {
            break;
        }
    }

    // Si se requiriese la historia, se podría retornar S_history;
    // en este ejemplo se retorna solo el S final.
    return S;
}


// Función para calcular la mediana de un vector
double median(std::vector<double>& data) {
    size_t n = data.size();
    std::sort(data.begin(), data.end());
    if (n % 2 == 0) {
        return (data[n / 2 - 1] + data[n / 2]) / 2.0;
    } else {
        return data[n / 2];
    }
}

void applyMovingMedianFilter(const std::vector<double>& signal, std::vector<double>& filtered, int window_size)
{
    filtered.resize(signal.size());
    int half = window_size / 2;

    for (size_t i = 0; i < signal.size(); ++i)
    {
        std::vector<double> window;
        for (int j = -half; j <= half; ++j)
        {
            int idx = i + j;
            if (idx >= 0 && idx < (int)signal.size())
                window.push_back(signal[idx]);
        }
        filtered[i] = median(window);
    }
}

// Función para calcular la MAD de un vector
double mad(const std::vector<double>& data, double median_value) {
    // Calcular las desviaciones absolutas respecto a la mediana
    std::vector<double> abs_deviations(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        abs_deviations[i] = std::abs(data[i] - median_value);
    }

    // Calcular la mediana de las desviaciones absolutas (MAD)
    return median(abs_deviations); // Escalar para ser consistente con la desviación estándar
}

// alpha < 1.0 para smoothing
void applyExponentialFilter(const std::vector<double>& signal, std::vector<double>& filtered, double alpha)
{
    filtered.resize(signal.size());
    filtered[0] = signal[0];

    for(size_t i = 1; i < signal.size(); ++i)
        filtered[i] = alpha * signal[i] + (1 - alpha) * filtered[i - 1];
}

// Función para calcular la mediana y la MAD de una ventana deslizante
void medMadDoubleTree(const std::vector<double>& in, std::vector<double>& out_med, std::vector<double>& out_mad, int window_size) {
    int n_samples = in.size();
    int half_window = window_size / 2;

    for (int i = 0; i < n_samples; ++i) {
        // Definir los límites de la ventana
        int start = std::max(0, i - half_window);
        int end = std::min(n_samples - 1, i + half_window);

        // Extraer la ventana actual
        std::vector<double> window(in.begin() + start, in.begin() + end + 1);

        // Calcular la mediana de la ventana
        out_med[i] = median(window);

        // Calcular la MAD de la ventana
        out_mad[i] = mad(window, out_med[i]);
    }
}

// Función para aplicar el filtro de Hampel
void hampelFilter(const std::vector<double>& data, std::vector<double>& result, int half_window_size, double nsigma) {
    int data_length = data.size();
    int window_size = 2 * half_window_size + 1;

    // Vectores para almacenar la mediana y la MAD
    std::vector<double> in_median(data_length);
    std::vector<double> MAD(data_length);

    // Calcular la mediana y la MAD para cada ventana
    medMadDoubleTree(data, in_median, MAD, window_size);

    // Aplicar el filtro de Hampel
    for (int i = 0; i < data_length; ++i) {
        double abs_deviation = std::abs(data[i] - in_median[i]);
        if (abs_deviation <= MAD[i] * 1.4826 * nsigma) {
            result[i] = data[i]; // Conservar el valor original
        } else {
            result[i] = in_median[i]; // Reemplazar con la mediana
        }
    }
}

// Función para aplicar el filtro de Hampel (variante)
void hampelFilterVariant(const std::vector<double>& data, std::vector<double>& result, int half_window_size, double nsigma) {
    int data_length = data.size();

    // Vectores para almacenar la mediana y la MAD
    std::vector<double> in_median(data_length);
    std::vector<double> MAD(data_length);

    // Calcular la mediana y la MAD para cada ventana
    for (int i = 0; i < data_length; ++i) {
        // Definir los límites de la ventana
        int start = std::max(0, i - half_window_size);
        int end = std::min(data_length - 1, i + half_window_size);

        // Extraer la ventana actual
        std::vector<double> window(data.begin() + start, data.begin() + end + 1);

        // Calcular la mediana de la ventana
        double window_median = median(window);

        // Calcular la MAD de la ventana
        double window_mad = mad(window, window_median);

        // Guardar los resultados
        in_median[i] = window_median;
        MAD[i] = window_mad;
    }

    // Aplicar la regla de decisión del filtro de Hampel
    for (int i = 0; i < data_length; ++i) {
        double abs_deviation = std::abs(data[i] - in_median[i]);
        if (abs_deviation <= MAD[i] * nsigma) {
            result[i] = data[i]; // Conservar el valor original
        } else {
            result[i] = in_median[i]; // Reemplazar con la mediana
        }
    }
}

// Agregador de ruido a señales 1D
std::vector<double> addNoiseAndSpikes(const std::vector<double>& clean_signal, double noise_level = 0.1, double spike_probability = 0.01, double spike_amplitude_max = 0.5)
{
    std::vector<double> noisy_signal = clean_signal;  // Copiar la señal limpia
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, noise_level);
    std::uniform_real_distribution<double> spike_prob_dist(0.0, 1.0);  // Probabilidad de que ocurra un spike
    std::uniform_real_distribution<double> spike_amplitude_dist(0.0, spike_amplitude_max); // Amplitud del spike

    // Añadir ruido y picos a la señal
    for (size_t i = 0; i < clean_signal.size(); ++i)
    {
        // Añadir ruido a la señal
        noisy_signal[i] += noise(generator);

        // Añadir spike con probabilidad especificada
        if (spike_prob_dist(generator) < spike_probability)
        {
            // Amplitud aleatoria del spike
            double spike_amplitude = spike_amplitude_dist(generator);
            noisy_signal[i] += spike_amplitude;  // Añadir spike con amplitud aleatoria
        }
    }
    return noisy_signal;
}

// Generar señal del acorde de Do mayor (C, E, G)
std::vector<double> generateChordSignal(double duration_ms, double sample_rate = 44100.0)
{
    int size = static_cast<int>((duration_ms / 1000.0) * sample_rate);

    // Frecuencias de las notas del acorde de Do Mayor (C, E, G)
    double freq_C = 261.63; // Do (C)
    double freq_E = 329.63; // Mi (E)
    double freq_G = 392.00; // Sol (G)

    std::vector<double> signal(size);

    for (int i = 0; i < size; ++i)
    {
        double t = i / sample_rate; // Tiempo en segundos

        // Sumar las señales seno de las frecuencias correspondientes
        signal[i] = std::sin(2 * M_PI * freq_C * t) + std::sin(2 * M_PI * freq_E * t) + std::sin(2 * M_PI * freq_G * t);
    }

    return signal;
}

void plotSignal1DMatplotlib(const std::vector<double>& signal, const std::vector<double>& xValues, const std::string& plotTitle) {
    plt::figure_size(1080, 720);
    plt::plot(xValues, signal, "r-");
    plt::title(plotTitle);
    plt::xlabel("Point");
    plt::ylabel("Value");
    plt::legend();
    plt::show();
}

// Función para calcular el MSE (Error Cuadrático Medio)
double calculateMSE(const std::vector<double>& original, const std::vector<double>& filtered)
{
    if (original.size() != filtered.size())
        throw std::invalid_argument("Las señales deben tener el mismo tamaño.");

    double mse = 0.0;
    for (size_t i = 0; i < original.size(); ++i)
    {
        double diff = original[i] - filtered[i];
        mse += diff * diff;
    }
    mse /= original.size();

    return mse;
}

// Función para calcular el PSNR (Relación Señal-Ruido de Pico)
double calculatePSNR(const std::vector<double>& original, const std::vector<double>& filtered, double max_value = 1.0)
{
    double mse = calculateMSE(original, filtered);
    if (mse == 0.0)
        return std::numeric_limits<double>::infinity(); // Si no hay error, PSNR es infinito
    double psnr = 10.0 * std::log10((max_value * max_value) / mse);
    return psnr;
}

int main()
{
    setenv("MPLBACKEND", "TkAgg", 1);

    // Creación perfil
    std::vector<double> clean_signal;
    std::vector<double> noisy_signal;
    std::vector<double> filtered_signal;

    bool useRealProfile = true; // true == usar perfil láser, false == usar acorde de DoM

    // Elegir método de filtrado
    enum FilteringMethods
    {
        TV = 0, //!< Filtro de variación total
        TV_EL,  //!< Filtro de variación total mediante Euler-Lagrange
        L0,     //!< L0 norm gradient minimization
        CHULL,  //!< Convex hull
        MA,     //!< Moving average
        MMA,    //!< Moving max
        MMI,    //!< Moving min
        MM,     //!< Moving median
        EF,     //!< Exponential filter
        H,      //!< Hampel
        HV,     //!< Hampel variant
        SG      //!< Savitzsky-Golay
    };

    FilteringMethods method{L0};

    // Leyendo
    if (useRealProfile)
    {
        // Perfil real

        cv::Mat cloud = cv::imread("/home/alejandro/Imágenes/water-blue-ocean-sea.jpg", cv::IMREAD_GRAYSCALE);
        if (cloud.channels() == 3) cv::extractChannel(cloud, cloud, 2);
        cloud.convertTo(cloud, CV_64F);
        std::cout << "cloud.type: " << cloud.type() << ", rows: " << cloud.rows << ", cols: " << cloud.cols << std::endl;

        if (cloud.empty())
        {
            std::cerr << "Error: El archivo de datos no se pudo cargar." << std::endl;
            return -1;
        }

        int profileNumber = 8;
        cv::Mat profile = cloud.row(profileNumber).clone();
        profile.convertTo(profile, CV_64F);
        clean_signal.assign((double*)profile.datastart, (double*)profile.dataend);
    }
    else
    {
        // Generar señal del acorde de Do mayor (C, E, G)

        double duration_ms = 10;      // Duración de la señal
        double sample_rate = 44100.0; // Frecuencia de muestreo en Hz
        clean_signal = generateChordSignal(duration_ms, sample_rate);
    }

    // Normalizar
    double max_value = *std::max_element(clean_signal.begin(), clean_signal.end());
    for (size_t i = 0; i < clean_signal.size(); ++i)
        clean_signal[i] /= max_value;

    // Añadir ruido
    double noise_level = 0.1;
    double spike_probability = 0.05;  // Probabilidad de un spike en cada punto
    double spike_amplitude_max = 0.25; // Amplitud máxima de los spikes
    noisy_signal = addNoiseAndSpikes(clean_signal, noise_level, spike_probability, spike_amplitude_max);

    // Inicializar señal resultado para algunos algoritmos
    filtered_signal = noisy_signal;

    // Componer abscisas
    std::vector<double> xValues(noisy_signal.size());
    for (size_t i = 0; i < xValues.size(); ++i)
        xValues[i] = static_cast<double>(i);

    // Guardar señal
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);
        std::string title = oss.str();
        plt::figure_size(1080, 720);
        plt::plot(xValues, clean_signal, {
                      {"label", "Clean"},
                      {"color", "g"},
                      {"lw", "1.5"}
                  });
        plt::plot(xValues, noisy_signal, {
                      {"label", "Noisy"},
                      {"color", "r"},
                      {"lw", "0.5"}
                  });
        plt::xlabel("Point");
        plt::ylabel("Value");
        plt::legend();
        std::string suffix_input = useRealProfile ? "profile" : "CMajor";
//        plt::save("/opt/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-1d-filtering/signal_" + suffix_input + ".png", 400);
        plt::show();
    }

    switch (method)
    {
        case TV:
        {
            double lambda = 2.0;    // Parámetro de fidelidad
            double step_size = 0.001; // Paso de tiempo
            int iterations = 50000;  // Número de iteraciones
            int n_iters = 1;
            for (int i = 1; i <= n_iters; ++i)
            {
                std::vector<double> input = filtered_signal;
                denoiseTV(filtered_signal, input, lambda, step_size, iterations);
            }
            break;
        }
        case TV_EL:
        {
            double lambda = 2.5;    // Parámetro de fidelidad
            double step_size = 0.001; // Paso de tiempo
            int iterations = 50000;  // Número de iteraciones
            int n_iters = 1;
            for (int i = 1; i <= n_iters; ++i)
            {
                std::vector<double> input = filtered_signal;
                denoiseTV_EL(filtered_signal, input, lambda, step_size, iterations);
            }
            break;
        }
        case L0:
        {
            double lambda = 0.05;
            double beta_max = 1e5;
            double kappa = 2.0;
            int max_iter = 50;
            int n_iters = 1;
            for (int i = 1; i <= n_iters; ++i)
            {
                std::vector<double> input = filtered_signal;
                filtered_signal = l0_gradient_minimization_1d(input, lambda, beta_max, kappa, max_iter);
            }
            break;
        }
        case CHULL:
        {
            convexHull1D(noisy_signal, filtered_signal);
            break;
        }
        case MA:
        {
            int window_size = 15;
            int n_iters = 1;
            for (int i = 1; i <= n_iters; ++i, window_size -= 3)
            {
                std::vector<double> temp_signal = filtered_signal;
                int current_window_size = std::max(1, window_size);
                applyMovingAverageFilter(temp_signal, filtered_signal, current_window_size);
            }
            break;
        }
        case MMA:
        {
            int window_size = 15;
            int n_iters = 1;
            for (int i = 1; i <= n_iters; ++i, window_size -= 3)
            {
                std::vector<double> temp_signal = filtered_signal;
                int current_window_size = std::max(1, window_size);
                applyMovingMaxFilter(temp_signal, filtered_signal, current_window_size);
            }
            break;
        }
        case MMI:
        {
            int window_size = 15;
            int n_iters = 1;
            for (int i = 1; i <= n_iters; ++i, window_size -= 3)
            {
                std::vector<double> temp_signal = filtered_signal;
                int current_window_size = std::max(1, window_size);
                applyMovingMinFilter(temp_signal, filtered_signal, current_window_size);
            }
            break;
        }
        case MM:
        {
            int window_size = 15;
            int n_iters = 1;
            for (int i = 1; i <= n_iters; ++i, window_size -= 3)
            {
                std::vector<double> temp_signal = filtered_signal;
                int current_window_size = std::max(1, window_size);
                applyMovingMedianFilter(temp_signal, filtered_signal, current_window_size);
            }
            break;
        }
        case EF:
        {
            double alpha = 0.25;
            int n_iters = 1;
            for (int i = 1; i <= n_iters; ++i)
            {
                std::vector<double> temp_signal = filtered_signal;
                applyExponentialFilter(temp_signal, filtered_signal, alpha);
            }
            break;
        }
        case H:
        {
            int halfSize = 7;
            double sigma = 0.25;
            int n_iters = 1;
            for (int i = 1; i <= n_iters; ++i)
            {
                std::vector<double> temp_signal = filtered_signal;
                hampelFilter(temp_signal, filtered_signal, halfSize, sigma);
            }
            break;
        }
        case HV:
        {
            int halfSize = 7;
            double sigma = 0.25;
            int n_iters = 1;
            for (int i = 1; i <= n_iters; ++i)
            {
                std::vector<double> temp_signal = filtered_signal;
                hampelFilterVariant(temp_signal, filtered_signal, halfSize, sigma);
            }
            break;
        }
        default:
        {
            break;
        }
    }

    std::string filter_label;
    switch (method)
    {
        case TV:
            filter_label = "TV";
            break;
        case TV_EL:
            filter_label = "TV_EL";
            break;
        case L0:
            filter_label = "L0";
            break;
        case CHULL:
            filter_label = "CHULL";
            break;
        case MA:
            filter_label = "MA";
            break;
        case MMA:
            filter_label = "MMA";
            break;
        case MMI:
            filter_label = "MMI";
            break;
        case MM:
            filter_label = "MM";
            break;
        case EF:
            filter_label = "EF";
            break;
        case H:
            filter_label = "H";
            break;
        case HV:
            filter_label = "HV";
            break;
        case SG:
            filter_label = "SG";
            break;
        default:
            filter_label = "Unknown";
            break;
    }

    // Análisis
    double mse = calculateMSE(clean_signal, filtered_signal);
    double psnr = calculatePSNR(clean_signal, filtered_signal);
    std::cout << "MSE: " << mse << std::endl;
    std::cout << "PSNR: " << psnr << " dB" << std::endl;

    // Graficar
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "Filtering with " << filter_label << " (MSE: " << mse << ", PSNR: " << psnr << " dB)";
    std::string title = oss.str();
    plt::figure_size(1080, 720);
    plt::plot(xValues, clean_signal, {
        {"label", "Clean"},
        {"color", "g"},
        {"lw", "0.5"}
    });
    plt::plot(xValues, noisy_signal, {
        {"label", "Noisy"},
        {"color", "r"},
        {"lw", "0.5"}
    });
    plt::plot(xValues, filtered_signal, {
        {"label", filter_label},
        {"color", "b"},
        {"lw", "1.0"}
    });
    plt::xlabel("Point");
    plt::ylabel("Value");
    plt::legend();
    std::string suffix_input = useRealProfile ? "profile" : "CMajor";
    plt::save("/opt/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-1d-filtering/" + filter_label + "_" + suffix_input + ".png", 400);
    plt::show();

    return 0;
}
