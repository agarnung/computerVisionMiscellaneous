#include <iostream>

#include <opencv4/opencv2/opencv.hpp>

#include <opencv2/opencv.hpp>
#include <vector>

void drawHistogram(
    const cv::Mat& hist,
    const std::string& outputPath,
    int hist_h,
    int bin_w,
    int bin_spacing,
    const cv::Scalar& evenBinColor,
    const cv::Scalar& oddBinColor,
    const cv::Scalar& backgroundColor,
    const cv::Scalar& borderColor,
    const cv::Scalar& gridColor,
    const cv::Scalar& percentileLineColor,
    double scaleFactor = 1.0
    )
{
    int hist_w = (bin_w + bin_spacing) * 256;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.4;
    int thickness = 1;
    int baseline = 0;
    cv::Size textSizeX = cv::getTextSize("255", fontFace, fontScale, thickness, &baseline);
    cv::Size textSizeY = cv::getTextSize("10000", fontFace, fontScale, thickness, &baseline);
    int margin_left = textSizeY.width + 10;
    int margin_bottom = textSizeX.height + 10;
    int margin_right = textSizeY.width / 2 + 10;
    int margin_top = textSizeX.height / 2 + 10;
    cv::Mat histImage(hist_h + margin_bottom + margin_top,
                      hist_w + margin_left + margin_right,
                      CV_8UC3, backgroundColor);
    cv::Mat histNorm;
    cv::normalize(hist, histNorm, 0, hist_h, cv::NORM_MINMAX);
    cv::line(histImage,
             cv::Point(margin_left, hist_h + margin_top),
             cv::Point(hist_w + margin_left, hist_h + margin_top),
             cv::Scalar(0, 0, 0), 1);
    cv::line(histImage,
             cv::Point(margin_left, margin_top),
             cv::Point(margin_left, hist_h + margin_top),
             cv::Scalar(0, 0, 0), 1);
    std::vector<float> percentiles = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    for (size_t i = 0; i < percentiles.size(); ++i)
    {
        int bin = cvRound(255 * percentiles[i]);
        int x_pos = margin_left + bin * (bin_w + bin_spacing);
        std::string label = std::to_string(bin);
        cv::putText(histImage, label,
                    cv::Point(x_pos - textSizeX.width / 2, hist_h + margin_top + textSizeX.height + 5),
                    fontFace, fontScale, cv::Scalar(0), thickness);
        cv::line(histImage,
                 cv::Point(x_pos, margin_top),
                 cv::Point(x_pos, hist_h + margin_top),
                 percentileLineColor, 1, cv::LINE_AA);
    }
    double histMax;
    cv::minMaxLoc(hist, nullptr, &histMax);
    for (size_t i = 0; i < percentiles.size(); ++i)
    {
        int value = cvRound(percentiles[i] * histMax);
        int y_pos = hist_h + margin_top - cvRound(percentiles[i] * hist_h);
        cv::putText(histImage, std::to_string(value),
                    cv::Point(5, y_pos + textSizeY.height / 2),
                    fontFace, fontScale, cv::Scalar(0), thickness);
        cv::line(histImage,
                 cv::Point(margin_left, y_pos),
                 cv::Point(hist_w + margin_left, y_pos),
                 percentileLineColor, 1, cv::LINE_AA);
    }
    for (int i = 0; i <= 256; i += 5)
    {
        int x_pos = margin_left + i * (bin_w + bin_spacing);
        cv::line(histImage,
                 cv::Point(x_pos, margin_top),
                 cv::Point(x_pos, hist_h + margin_top),
                 gridColor, 1, cv::LINE_AA);
    }
    for (int i = 0; i <= hist_h; i += hist_h / 10)
    {
        cv::line(histImage,
                 cv::Point(margin_left, hist_h + margin_top - i),
                 cv::Point(hist_w + margin_left, hist_h + margin_top - i),
                 gridColor, 1, cv::LINE_AA);
    }
    for (int i = 0; i < 256; i++)
    {
        int binValue = cvRound(histNorm.at<float>(i));
        int x_start = margin_left + i * (bin_w + bin_spacing);
        cv::Scalar binColor = (i % 2 == 0) ? evenBinColor : oddBinColor;
        cv::rectangle(histImage,
                      cv::Point(x_start, hist_h + margin_top),
                      cv::Point(x_start + bin_w, hist_h + margin_top - binValue),
                      binColor, cv::FILLED);
        cv::rectangle(histImage,
                      cv::Point(x_start, hist_h + margin_top),
                      cv::Point(x_start + bin_w, hist_h + margin_top - binValue),
                      borderColor, 1);
    }
    if (scaleFactor > 0.0)
        cv::resize(histImage, histImage, cv::Size(0, 0), scaleFactor, scaleFactor, cv::INTER_NEAREST);
    if (!outputPath.empty())
        cv::imwrite(outputPath, histImage);
    cv::imshow(outputPath, histImage);
    cv::waitKey(0);
}

void drawHistogram2(
    const cv::Mat& hist,
    const std::string& outputPath,
    int hist_h,
    int bin_w,
    int bin_spacing,
    const cv::Scalar& evenBinColor,
    const cv::Scalar& oddBinColor,
    const cv::Scalar& backgroundColor,
    const cv::Scalar& borderColor,
    const cv::Scalar& gridColor,
    const cv::Scalar& percentileLineColor,
    double scaleFactor = 1.0,
    const std::string& title = "",
    const std::string& xLabel = "Intensidad",
    const std::string& yLabel = "Frecuencia"
    ) {
    int hist_w = (bin_w + bin_spacing) * 256;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.4;
    int thickness = 1;
    int baseline = 0;
    cv::Size textSizeX = cv::getTextSize("255", fontFace, fontScale, thickness, &baseline);
    cv::Size textSizeY = cv::getTextSize("10000", fontFace, fontScale, thickness, &baseline);

    int margin_left = textSizeY.width + 30;
    int margin_bottom = textSizeX.height + 30;
    int margin_right = textSizeY.width / 2 + 30;
    int margin_top = textSizeX.height / 2 + 40;

    cv::Mat histImage(hist_h + margin_bottom + margin_top,
                      hist_w + margin_left + margin_right,
                      CV_8UC3, backgroundColor);
    cv::Mat histNorm;
    cv::normalize(hist, histNorm, 0, hist_h, cv::NORM_MINMAX);

    for (int i = 0; i <= 256; i += 5) {
        int x_pos = margin_left + i * (bin_w + bin_spacing);
        cv::line(histImage,
                 cv::Point(x_pos, margin_top),
                 cv::Point(x_pos, hist_h + margin_top),
                 gridColor, 1, cv::LINE_AA);
    }
    for (int i = 0; i <= hist_h; i += hist_h / 10) {
        cv::line(histImage,
                 cv::Point(margin_left, hist_h + margin_top - i),
                 cv::Point(hist_w + margin_left, hist_h + margin_top - i),
                 gridColor, 1, cv::LINE_AA);
    }

    for (int i = 0; i < 256; i++) {
        int binValue = cvRound(histNorm.at<float>(i));
        int x_start = margin_left + i * (bin_w + bin_spacing);
        cv::Scalar binColor = (i % 2 == 0) ? evenBinColor : oddBinColor;
        cv::rectangle(histImage,
                      cv::Point(x_start, hist_h + margin_top),
                      cv::Point(x_start + bin_w, hist_h + margin_top - binValue),
                      binColor, cv::FILLED);
        cv::rectangle(histImage,
                      cv::Point(x_start, hist_h + margin_top),
                      cv::Point(x_start + bin_w, hist_h + margin_top - binValue),
                      borderColor, thickness);
    }

    cv::line(histImage,
             cv::Point(margin_left, hist_h + margin_top),
             cv::Point(hist_w + margin_left, hist_h + margin_top),
             borderColor, thickness);
    cv::line(histImage,
             cv::Point(margin_left, margin_top),
             cv::Point(margin_left, hist_h + margin_top),
             borderColor, thickness);

    if (!xLabel.empty()) {
        cv::Size xLabelSize = cv::getTextSize(xLabel, fontFace, fontScale, thickness, &baseline);
        int xLabelX = margin_left + (hist_w / 2) - (xLabelSize.width / 2);
        int xLabelY = hist_h + margin_top + xLabelSize.height + 20;
        cv::putText(histImage, xLabel,
                    cv::Point(xLabelX, xLabelY),
                    fontFace, fontScale, borderColor, thickness);
    }

    if (!yLabel.empty()) {
        cv::Mat rotatedLabel;
        cv::Mat labelMat = cv::Mat::zeros(textSizeY.height + 20, textSizeY.width + 20, CV_8UC3);
        labelMat.setTo(backgroundColor);
        cv::putText(labelMat, yLabel,
                    cv::Point(10, textSizeY.height + 10),
                    fontFace, fontScale, borderColor, thickness);
        cv::rotate(labelMat, rotatedLabel, cv::ROTATE_90_COUNTERCLOCKWISE);

        int yLabelX = margin_left - 90;
        int yLabelY = margin_top + (hist_h / 2) - (rotatedLabel.rows / 2);
        rotatedLabel.copyTo(histImage(cv::Rect(yLabelX, yLabelY, rotatedLabel.cols, rotatedLabel.rows)));
    }

    if (!title.empty()) {
        double titleFontScale = 0.5; // Aumentar el tamaño de la fuente
        int titleThickness = 2;   // Aumentar el grosor del texto
        cv::Size titleSize = cv::getTextSize(title, fontFace, titleFontScale, titleThickness, &baseline);
        int titleX = margin_left + (hist_w / 2) - (titleSize.width / 2); // Centrar el título
        int titleY = margin_top - 10; // Margen superior
        cv::putText(histImage, title,
                    cv::Point(titleX, titleY),
                    fontFace, titleFontScale, borderColor, titleThickness);
    }
    std::vector<float> percentiles = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    for (size_t i = 0; i < percentiles.size(); ++i) {
        int bin = cvRound(255 * percentiles[i]);
        int x_pos = margin_left + bin * (bin_w + bin_spacing);
        std::string label = std::to_string(bin);
        cv::putText(histImage, label,
                    cv::Point(x_pos - textSizeX.width / 2, hist_h + margin_top + textSizeX.height + 5),
                    fontFace, fontScale, borderColor, thickness);
        cv::line(histImage,
                 cv::Point(x_pos, margin_top),
                 cv::Point(x_pos, hist_h + margin_top),
                 percentileLineColor, thickness, cv::LINE_AA);
    }

    double histMax;
    cv::minMaxLoc(hist, nullptr, &histMax);
    for (size_t i = 0; i < percentiles.size(); ++i) {
        int value = cvRound(percentiles[i] * histMax);
        int y_pos = hist_h + margin_top - cvRound(percentiles[i] * hist_h);
        cv::putText(histImage, std::to_string(value),
                    cv::Point(5, y_pos + textSizeY.height / 2),
                    fontFace, fontScale, borderColor, thickness);
        cv::line(histImage,
                 cv::Point(margin_left, y_pos),
                 cv::Point(hist_w + margin_left, y_pos),
                 percentileLineColor, thickness, cv::LINE_AA);
    }

    if (scaleFactor > 0.0)
        cv::resize(histImage, histImage, cv::Size(0, 0), scaleFactor, scaleFactor, cv::INTER_LANCZOS4);

    if (!outputPath.empty())
        cv::imwrite(outputPath, histImage);
    cv::imshow(outputPath, histImage);
    cv::waitKey(0);
}


int main()
{
    // Leer la imagen en escala de grises
    cv::Mat img = cv::imread("/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/lena-eyes.png", cv::IMREAD_GRAYSCALE);
    if (img.empty() || !img.isContinuous())
    {
        std::cerr << "Error loading image." << std::endl;
        return -1;
    }
    cv::imshow("lena-eyes", img);
    cv::waitKey(0);

    // Calcular el histograma
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    std::cout << hist.t() << std::endl;

    // Representar el histograma como una señal 1D
    cv::Mat hist1D(1, 256, CV_32F);
    cv::normalize(hist, hist1D, 0, 1, cv::NORM_MINMAX);
    hist1D = hist1D.t();
    cv::imwrite("/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/hist1D.png", hist1D);
    cv::imshow("1D Histograma", hist1D * 255);
    cv::waitKey(0);

    // Gráfica el histograma definiendo altura (cada bin mide 1 pixel de ancho, no hace definir altura ni ancho del bin)=
    {
        const int hist_h = 200;
        cv::Mat histImage(hist_h, 256, CV_8UC1, cv::Scalar(255));
        cv::Mat hist_norm;
        cv::normalize(hist, hist_norm, 0, hist_h, cv::NORM_MINMAX);
        for (int i = 0; i < histSize; i++)
        {
            cv::line(histImage,
                     cv::Point(i, hist_h - cvRound(hist_norm.at<float>(i - 1))),
                     cv::Point(i, hist_h - cvRound(hist_norm.at<float>(i))),
                     cv::Scalar(0), 1);
        }
        cv::imwrite("/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/hist2D.png", histImage);
        cv::imshow("Histogram", histImage);
        cv::waitKey(0);
    }

    // Gráfica el histograma definiendo altura y anchura
    {
        const int hist_w = 512, hist_h = 400;
        const int bin_w = cvRound((double)hist_w / histSize);
        cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(255));
        cv::Mat hist_norm;
        cv::normalize(hist, hist_norm, 0, histImage.rows, cv::NORM_MINMAX);
        for (int i = 1; i < histSize; i++)
        {
            cv::line(histImage,
                     cv::Point(bin_w * (i - 1), hist_h - cvRound(hist_norm.at<float>(i - 1))),
                     cv::Point(bin_w * i, hist_h - cvRound(hist_norm.at<float>(i))),
                     cv::Scalar(0), 2, cv::LINE_AA | cv::FILLED, 0);
        }
        cv::imwrite("/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/hist2D_2.png", histImage);
        cv::imshow("Histogram 2", histImage);
        cv::waitKey(0);
    }

    // Gráfica el histograma definiendo altura y ancho de bin
    {
        int hist_h = 400;
        int bin_w = 3;
        int bin_spacing = 2;
        int hist_w = (bin_w + bin_spacing) * 256;
        cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(255));
        cv::Mat histNorm;
        cv::normalize(hist, histNorm, 0, hist_h, cv::NORM_MINMAX);
        for (int i = 0; i < histSize; i++) {
            int binValue = cvRound(histNorm.at<float>(i));
            int x_start = i * (bin_w + bin_spacing);
            cv::rectangle(histImage,
                          cv::Point(x_start, hist_h),
                          cv::Point(x_start + bin_w, hist_h - binValue),
                          cv::Scalar(0), cv::FILLED);
        }
        cv::imwrite("/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/hist2D_3.png", histImage);
        cv::imshow("Histogram 3", histImage);
        cv::waitKey(0);
    }

    // Añadir etiquetas
    {
        int hist_h = 400;
        int bin_w = 3;
        int bin_spacing = 2;
        int hist_w = (bin_w + bin_spacing) * 256;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.4;
        int thickness = 1;
        int baseline = 0;
        cv::Size textSizeX = cv::getTextSize("255", fontFace, fontScale, thickness, &baseline);
        cv::Size textSizeY = cv::getTextSize("10000", fontFace, fontScale, thickness, &baseline);
        int margin_left = textSizeY.width + 10;
        int margin_bottom = textSizeX.height + 10;
        int margin_right = textSizeY.width / 2 + 10;
        int margin_top = textSizeX.height / 2 + 10;
        cv::Mat histImage(hist_h + margin_bottom + margin_top,
                          hist_w + margin_left + margin_right,
                          CV_8UC1, cv::Scalar(255));
        cv::Mat histNorm;
        cv::normalize(hist, histNorm, 0, hist_h, cv::NORM_MINMAX);
        for (int i = 0; i < 256; i++)
        {
            int binValue = cvRound(histNorm.at<float>(i));
            int x_start = margin_left + i * (bin_w + bin_spacing);

            cv::rectangle(histImage,
                          cv::Point(x_start, hist_h + margin_top),
                          cv::Point(x_start + bin_w, hist_h + margin_top - binValue),
                          cv::Scalar(0), cv::FILLED);
        }
        for (int i = 0; i <= 256; i += 128)
        {
            int x_pos = margin_left + i * (bin_w + bin_spacing);
            cv::putText(histImage, std::to_string(i),
                        cv::Point(x_pos - textSizeX.width / 2, hist_h + margin_top + textSizeX.height + 5),
                        fontFace, fontScale, cv::Scalar(0), thickness);
        }
        double histMax;
        cv::minMaxLoc(hist, nullptr, &histMax);
        for (int i = 0; i <= hist_h; i += hist_h / 2)
        {
            int value = cvRound(((double)i / hist_h) * histMax);

            cv::putText(histImage, std::to_string(value),
                        cv::Point(5, hist_h + margin_top - i + textSizeY.height / 2),
                        fontFace, fontScale, cv::Scalar(0), thickness);
        }
        cv::imwrite("/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/hist2D_4.png", histImage);
        cv::imshow("Histogram 4", histImage);
        cv::waitKey(0);
    }

    // Ejes coordenados
    {
        int hist_h = 400;
        int bin_w = 3;
        int bin_spacing = 2;
        int hist_w = (bin_w + bin_spacing) * 256;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.4;
        int thickness = 1;
        int baseline = 0;
        cv::Size textSizeX = cv::getTextSize("255", fontFace, fontScale, thickness, &baseline);
        cv::Size textSizeY = cv::getTextSize("10000", fontFace, fontScale, thickness, &baseline);
        int margin_left = textSizeY.width + 10;
        int margin_bottom = textSizeX.height + 10;
        int margin_right = textSizeY.width / 2 + 10;
        int margin_top = textSizeX.height / 2 + 10;
        cv::Mat histImage(hist_h + margin_bottom + margin_top,
                          hist_w + margin_left + margin_right,
                          CV_8UC1, cv::Scalar(255));
        cv::Mat histNorm;
        cv::normalize(hist, histNorm, 0, hist_h, cv::NORM_MINMAX);
        for (int i = 0; i < 256; i++)
        {
            int binValue = cvRound(histNorm.at<float>(i));
            int x_start = margin_left + i * (bin_w + bin_spacing);

            cv::rectangle(histImage,
                          cv::Point(x_start, hist_h + margin_top),
                          cv::Point(x_start + bin_w, hist_h + margin_top - binValue),
                          cv::Scalar(0), cv::FILLED);
        }
        cv::line(histImage,
                 cv::Point(margin_left, hist_h + margin_top),
                 cv::Point(hist_w + margin_left, hist_h + margin_top),
                 cv::Scalar(0), 1);
        cv::line(histImage,
                 cv::Point(margin_left, margin_top),
                 cv::Point(margin_left, hist_h + margin_top),
                 cv::Scalar(0), 1);
        for (int i = 0; i <= 256; i += 128)
        {
            int x_pos = margin_left + i * (bin_w + bin_spacing);
            cv::putText(histImage, std::to_string(i),
                        cv::Point(x_pos - textSizeX.width / 2, hist_h + margin_top + textSizeX.height + 5),
                        fontFace, fontScale, cv::Scalar(0), thickness);
        }
        double histMax;
        cv::minMaxLoc(hist, nullptr, &histMax);
        for (int i = 0; i <= hist_h; i += hist_h / 2)
        {
            int value = cvRound(((double)i / hist_h) * histMax);

            cv::putText(histImage, std::to_string(value),
                        cv::Point(5, hist_h + margin_top - i + textSizeY.height / 2),
                        fontFace, fontScale, cv::Scalar(0), thickness);
        }
        cv::imwrite("/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/hist2D_5.png", histImage);
        cv::imshow("Histogram 5", histImage);
        cv::waitKey(0);
    }

    // Añadir color
    {
        int hist_h = 400;
        int bin_w = 3;
        int bin_spacing = 2;
        int hist_w = (bin_w + bin_spacing) * 256;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.4;
        int thickness = 1;
        int baseline = 0;
        cv::Scalar evenBinColor(0, 0, 255);
        cv::Scalar oddBinColor(0, 255, 0);
        cv::Scalar backgroundColor(240, 241, 240);
        cv::Scalar borderColor(0, 0, 0);
        cv::Size textSizeX = cv::getTextSize("255", fontFace, fontScale, thickness, &baseline);
        cv::Size textSizeY = cv::getTextSize("10000", fontFace, fontScale, thickness, &baseline);
        int margin_left = textSizeY.width + 10;
        int margin_bottom = textSizeX.height + 10;
        int margin_right = textSizeY.width / 2 + 10;
        int margin_top = textSizeX.height / 2 + 10;
        cv::Mat histImage(hist_h + margin_bottom + margin_top,
                          hist_w + margin_left + margin_right,
                          CV_8UC3, backgroundColor);
        cv::Mat histNorm;
        cv::normalize(hist, histNorm, 0, hist_h, cv::NORM_MINMAX);
        for (int i = 0; i < 256; i++)
        {
            int binValue = cvRound(histNorm.at<float>(i));
            int x_start = margin_left + i * (bin_w + bin_spacing);
            cv::Scalar binColor = (i % 2 == 0) ? evenBinColor : oddBinColor;
            cv::rectangle(histImage,
                          cv::Point(x_start, hist_h + margin_top),
                          cv::Point(x_start + bin_w, hist_h + margin_top - binValue),
                          binColor, cv::FILLED);
            cv::rectangle(histImage,
                          cv::Point(x_start, hist_h + margin_top),
                          cv::Point(x_start + bin_w, hist_h + margin_top - binValue),
                          borderColor, 1);
        }
        cv::line(histImage,
                 cv::Point(margin_left, hist_h + margin_top),
                 cv::Point(hist_w + margin_left, hist_h + margin_top),
                 cv::Scalar(0, 0, 0), 1);
        cv::line(histImage,
                 cv::Point(margin_left, margin_top),
                 cv::Point(margin_left, hist_h + margin_top),
                 cv::Scalar(0, 0, 0), 1);
        for (int i = 0; i <= 256; i += 128)
        {
            int x_pos = margin_left + i * (bin_w + bin_spacing);
            cv::putText(histImage, std::to_string(i),
                        cv::Point(x_pos - textSizeX.width / 2, hist_h + margin_top + textSizeX.height + 5),
                        fontFace, fontScale, cv::Scalar(0), thickness);
        }
        double histMax;
        cv::minMaxLoc(hist, nullptr, &histMax);
        for (int i = 0; i <= hist_h; i += hist_h / 2)
        {
            int value = cvRound(((double)i / hist_h) * histMax);
            cv::putText(histImage, std::to_string(value),
                        cv::Point(5, hist_h + margin_top - i + textSizeY.height / 2),
                        fontFace, fontScale, cv::Scalar(0), thickness);
        }
        cv::imwrite("/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/hist2D_6.png", histImage);
        cv::imshow("Histogram 6", histImage);
        cv::waitKey(0);
    }

    // Lineas punteadas y etiquetas percentiles
    {
        int hist_h = 400;
        int bin_w = 3;
        int bin_spacing = 2;
        int hist_w = (bin_w + bin_spacing) * 256;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.4;
        int thickness = 1;
        int baseline = 0;
        cv::Scalar evenBinColor(0, 0, 255);
        cv::Scalar oddBinColor(0, 255, 0);
        cv::Scalar backgroundColor(240, 241, 240);
        cv::Scalar borderColor(0, 0, 0);
        cv::Scalar gridColor(200, 200, 200);
        cv::Scalar percentileLineColor(150, 150, 150);
        cv::Size textSizeX = cv::getTextSize("255", fontFace, fontScale, thickness, &baseline);
        cv::Size textSizeY = cv::getTextSize("10000", fontFace, fontScale, thickness, &baseline);
        int margin_left = textSizeY.width + 10;
        int margin_bottom = textSizeX.height + 10;
        int margin_right = textSizeY.width / 2 + 10;
        int margin_top = textSizeX.height / 2 + 10;
        cv::Mat histImage(hist_h + margin_bottom + margin_top,
                          hist_w + margin_left + margin_right,
                          CV_8UC3, backgroundColor);
        cv::Mat histNorm;
        cv::normalize(hist, histNorm, 0, hist_h, cv::NORM_MINMAX);
        for (int i = 0; i < 256; i++)
        {
            int binValue = cvRound(histNorm.at<float>(i));
            int x_start = margin_left + i * (bin_w + bin_spacing);
            cv::Scalar binColor = (i % 2 == 0) ? evenBinColor : oddBinColor;
            cv::rectangle(histImage,
                          cv::Point(x_start, hist_h + margin_top),
                          cv::Point(x_start + bin_w, hist_h + margin_top - binValue),
                          binColor, cv::FILLED);
            cv::rectangle(histImage,
                          cv::Point(x_start, hist_h + margin_top),
                          cv::Point(x_start + bin_w, hist_h + margin_top - binValue),
                          borderColor, 1);
        }
        cv::line(histImage,
                 cv::Point(margin_left, hist_h + margin_top),
                 cv::Point(hist_w + margin_left, hist_h + margin_top),
                 cv::Scalar(0, 0, 0), 1);
        cv::line(histImage,
                 cv::Point(margin_left, margin_top),
                 cv::Point(margin_left, hist_h + margin_top),
                 cv::Scalar(0, 0, 0), 1);
        std::vector<float> percentiles = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
        for (size_t i = 0; i < percentiles.size(); ++i)
        {
            int bin = cvRound(255 * percentiles[i]);
            int x_pos = margin_left + bin * (bin_w + bin_spacing);
            std::string label = std::to_string(bin);
            cv::putText(histImage, label,
                        cv::Point(x_pos - textSizeX.width / 2, hist_h + margin_top + textSizeX.height + 5),
                        fontFace, fontScale, cv::Scalar(0), thickness);
            cv::line(histImage,
                     cv::Point(x_pos, margin_top),
                     cv::Point(x_pos, hist_h + margin_top),
                     percentileLineColor, 1, cv::LINE_AA);
        }
        double histMax;
        cv::minMaxLoc(hist, nullptr, &histMax);
        for (size_t i = 0; i < percentiles.size(); ++i)
        {
            int value = cvRound(percentiles[i] * histMax);
            int y_pos = hist_h + margin_top - cvRound(percentiles[i] * hist_h);
            cv::putText(histImage, std::to_string(value),
                        cv::Point(5, y_pos + textSizeY.height / 2),
                        fontFace, fontScale, cv::Scalar(0), thickness);
            cv::line(histImage,
                     cv::Point(margin_left, y_pos),
                     cv::Point(hist_w + margin_left, y_pos),
                     percentileLineColor, 1, cv::LINE_AA);
        }

        for (int i = 0; i <= 256; i += 5)
        {
            int x_pos = margin_left + i * (bin_w + bin_spacing);
            cv::line(histImage,
                     cv::Point(x_pos, margin_top),
                     cv::Point(x_pos, hist_h + margin_top),
                     gridColor, 1, cv::LINE_AA);
        }
        for (int i = 0; i <= hist_h; i += hist_h / 10)
        {
            cv::line(histImage,
                     cv::Point(margin_left, hist_h + margin_top - i),
                     cv::Point(hist_w + margin_left, hist_h + margin_top - i),
                     gridColor, 1, cv::LINE_AA);
        }
        cv::imwrite("/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/hist2D_7.png", histImage);
        cv::imshow("Histogram 7", histImage);
        cv::waitKey(0);
    }

    // Función
    {
        drawHistogram(
            hist,
            "/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/hist2D_8.png",
            800,                       // hist_h
            3,                         // bin_w
            2,                         // bin_spacing
            cv::Scalar(0, 0, 255),     // evenBinColor
            cv::Scalar(0, 255, 0),     // oddBinColor
            cv::Scalar(240, 241, 240), // backgroundColor
            cv::Scalar(0, 0, 0),       // borderColor
            cv::Scalar(200, 200, 200), // gridColor
            cv::Scalar(150, 150, 150), // percentileLineColor
            1.0                        // scaleFactor
            );
    }

    // Función
    {
        drawHistogram2(
            hist,
            "/opt/proyectos/unpublished-posts/unpublised_data/unpublised_images/XXXX-XX-XX-crafting-my-histogram/hist2D_9.png",
            400, // hist_h
            3,   // bin_w
            2,   // bin_spacing
            cv::Scalar(0, 0, 255), // evenBinColor
            cv::Scalar(0, 255, 0), // oddBinColor
            cv::Scalar(240, 241, 240), // backgroundColor
            cv::Scalar(0, 0, 0), // borderColor
            cv::Scalar(200, 200, 200), // gridColor
            cv::Scalar(150, 150, 150), // percentileLineColor
            1.0, // scaleFactor
            "Histograma de Intensidad", // título
            "Intensidad de pixeles", // leyenda del eje X
            "Frecuencia" // leyenda del eje Y
            );
    }

    return 0;
}
