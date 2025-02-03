#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

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

void plotSignal1DGnuplot(
    const std::vector<double>& g,
    const std::string& plotTitle,
    const std::string& backgroundColor = "#FFFFFF",
    const std::string& signalColor = "blue",
    const std::string& graphTitle = "Graph"
    ) {
    if (g.empty()) {
        std::cerr << "El vector está vacío!" << std::endl;
        return;
    }

    FILE* gnuplotPipe = popen("gnuplot -p", "w");
    if (!gnuplotPipe) {
        std::cerr << "Error al abrir el pipe a gnuplot!" << std::endl;
        return;
    }

    fprintf(gnuplotPipe, "set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb '%s' behind\n", backgroundColor.c_str());
    fprintf(gnuplotPipe, "set title '%s'\n", graphTitle.c_str());
    fprintf(gnuplotPipe, "set style line 1 linecolor rgb '%s' linetype 1 linewidth 2\n", signalColor.c_str());
    fprintf(gnuplotPipe, "plot '-' with lines title '%s' ls 1\n", plotTitle.c_str());
    for (int i = 0; i < g.size(); ++i) {
        fprintf(gnuplotPipe, "%d %f\n", i, g[i]);
    }
    fprintf(gnuplotPipe, "e\n");

    fclose(gnuplotPipe);
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

void plotMultipleSignals1DGnuplot(
    const std::vector<std::vector<double>>& signals,
    const std::vector<std::string>& plotTitles,
    const std::vector<std::string>& signalColors,
    const std::string& backgroundColor = "#FFFFFF",
    const std::string& graphTitle = "Graph"
    ) {
    if (signals.empty()) {
        std::cerr << "No hay señales para graficar!" << std::endl;
        return;
    }
    if (signals.size() != plotTitles.size() || signals.size() != signalColors.size()) {
        std::cerr << "El número de señales, títulos y colores debe ser el mismo!" << std::endl;
        return;
    }

    FILE* gnuplotPipe = popen("gnuplot -p", "w");
    if (!gnuplotPipe) {
        std::cerr << "Error al abrir el pipe a gnuplot!" << std::endl;
        return;
    }

    fprintf(gnuplotPipe, "set object 1 rectangle from screen 0,0 to screen 1,1 fillcolor rgb '%s' behind\n", backgroundColor.c_str());
    fprintf(gnuplotPipe, "set title '%s'\n", graphTitle.c_str());

    for (size_t i = 0; i < signals.size(); ++i) {
        fprintf(gnuplotPipe, "set style line %zu linecolor rgb '%s' linetype 1 linewidth 2\n", i + 1, signalColors[i].c_str());
    }

    fprintf(gnuplotPipe, "plot ");
    for (size_t i = 0; i < signals.size(); ++i) {
        if (i > 0) {
            fprintf(gnuplotPipe, ", ");
        }
        fprintf(gnuplotPipe, "'-' with lines title '%s' ls %zu", plotTitles[i].c_str(), i + 1);
    }
    fprintf(gnuplotPipe, "\n");

    for (size_t i = 0; i < signals.size(); ++i) {
        for (size_t j = 0; j < signals[i].size(); ++j) {
            fprintf(gnuplotPipe, "%zu %f\n", j, signals[i][j]);
        }
        fprintf(gnuplotPipe, "e\n");
    }

    fclose(gnuplotPipe);
}

int main()
{
    setenv("MPLBACKEND", "TkAgg", 1);

    cv::Mat cloud = cv::imread("/media/sf_shared_folder/blurred-vision-678x446_compressed.jpg", cv::IMREAD_GRAYSCALE);
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
    std::vector<double> noisy_signal;
    for (int i = 0; i < profile.cols; ++i)
        noisy_signal.push_back(profile.at<double>(0, i));

    std::vector<double> xValues(noisy_signal.size());
    for (size_t i = 0; i < xValues.size(); ++i)
        xValues[i] = static_cast<double>(i);
    plt::figure_size(1080, 720);
    plt::plot(xValues, noisy_signal, "r-");
    plt::title("Original profile with matplotlibcpp");
    plt::xlabel("Point");
    plt::ylabel("Value");
    plt::show();

    int window_size = 15;
    std::vector<double> smooth_signal = noisy_signal;
    for (int i = 1; i <= 3; ++i, window_size -= 3)
    {
        std::vector<double> temp_signal = smooth_signal;
        int current_window_size = std::max(1, window_size);
        applyMovingMaxFilter(temp_signal, smooth_signal, current_window_size);
    }

    for (size_t i = 0; i < xValues.size(); ++i)
        xValues[i] = static_cast<double>(i);
    plt::figure_size(1080, 720);
    plt::plot(xValues, noisy_signal, {{"label", "Noisy"}, {"color", "r"}});
    plt::plot(xValues, smooth_signal, {{"label", "Filtered"}, {"color", "b"}});
    plt::title("Comparison with matplotlibcpp");
    plt::xlabel("Point");
    plt::ylabel("Value");
    plt::legend();
    plt::show();

    plotSignal1DGnuplot(smooth_signal, "Filtered signal with gnuplot", "#FFFFFF", "blue");

    std::vector<std::vector<double>> signals = {noisy_signal, smooth_signal};
    std::vector<std::string> titles = {"Signal 1", "Signal 2"};
    std::vector<std::string> colors = {"red", "green"};
    plotMultipleSignals1DGnuplot(signals, titles, colors, "#222222", "Comparison with gnuplot");

    return 0;
}
