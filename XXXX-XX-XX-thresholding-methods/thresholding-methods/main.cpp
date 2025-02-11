/**
 * Distintos algoritmos para umbralizar imágenes, principalmente para detectar defectos de un defect-map
 **/

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <common/cv/matrixfunctions.h>
#include <common/cv/matlabfunctions.h>

void muestraImagenOpenCV(const cv::Mat img, std::string title, bool destroyAfter)
{
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::resizeWindow(title, 800, 600);
    cv::imshow(title, img);
    cv::waitKey(0);

    if (destroyAfter)
        cv::destroyWindow(title);
}

static void onTrackbarChange(int thresholdValue, void* userData)
{
    cv::Mat* inputImage = static_cast<cv::Mat*>(userData);
    cv::Mat binaryImage;
    cv::threshold(*inputImage, binaryImage, thresholdValue, 255, cv::THRESH_BINARY);
    cv::imshow("Binary Image", binaryImage);
}

cv::Mat triangleThreshold(const cv::Mat& image)
{
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&image, 1, nullptr, cv::noArray(), hist, 1, &histSize, &histRange);

    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(hist, &min_val, &max_val, &min_loc, &max_loc);
    bool flipea = max_loc.y - 0.0 < 255.0 - max_loc.y;
    if (flipea)
    {
        cv::flip(hist, hist, 0);
        cv::minMaxLoc(hist, &min_val, &max_val, &min_loc, &max_loc);
        std::cout << "triangleThreshold - se hizo flip" << std::endl;
    }
    std::cout << "triangleThreshold - min_loc.x: " << min_loc.x << std::endl;
    std::cout << "triangleThreshold - min_loc.y: " << min_loc.y << std::endl;
    std::cout << "triangleThreshold - max_loc.x: " << max_loc.x << std::endl;
    std::cout << "triangleThreshold - max_loc.y: " << max_loc.y << std::endl;
    std::cout << "triangleThreshold - min_val: " << min_val << std::endl;
    std::cout << "triangleThreshold - max_val: " << max_val << std::endl;

    int threshold_val = 0.0;
    double maxLength = std::numeric_limits<double>::min();
    for (int bin = 1; bin < max_loc.y; ++bin)
    {
        if (cvRound(hist.at<float>(bin)) == 0)
            continue;

        cv::Mat hist_image(400, 512, CV_8UC3, cv::Scalar(255, 255, 255));
        int bin_width = hist_image.cols / histSize;
        cv::normalize(hist, hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());

        for (int i = 1; i < histSize; ++i)
        {
            cv::line(hist_image,
                     cv::Point(bin_width * (i - 1), hist_image.rows - cvRound(hist.at<float>(i - 1))),
                     cv::Point(bin_width * i, hist_image.rows - cvRound(hist.at<float>(i))),
                     cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        }

        cv::Point p1MinMax(bin_width * min_loc.y, hist_image.rows - cvRound(hist.at<float>(min_loc.y)));
        cv::Point p2MinMax(bin_width * max_loc.y, hist_image.rows - cvRound(hist.at<float>(max_loc.y)));
        cv::line(hist_image,
                 p1MinMax,
                 p2MinMax,
                 cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        double slope = (p1MinMax.y - p2MinMax.y) / (p1MinMax.x - p2MinMax.x);

        double perpendicular_slope = -1 / slope;
        double bin_center_x = bin_width * (bin + 0.5);
        double bin_center_y = hist_image.rows - cvRound(hist.at<float>(bin)) - 0.5 * hist_image.rows / histSize;
        double intercept_blue = p1MinMax.y - slope * p1MinMax.x;
        double intercept_red = bin_center_y - perpendicular_slope * bin_center_x;
        int x_intersection = static_cast<int>((intercept_red - intercept_blue) / (slope - perpendicular_slope));
        int y_intersection = static_cast<int>(slope * x_intersection + intercept_blue);
        cv::Point pTh(bin_center_x, bin_center_y);
        cv::Point intersection_point(x_intersection - std::exp(bin_width * 0.009 * bin), y_intersection - std::exp(bin_width * 0.009 * bin));
        cv::line(hist_image, pTh, intersection_point, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        cv::line(hist_image, cv::Point(bin_center_x, hist_image.rows), cv::Point(bin_center_x, 0), cv::Scalar(255, 0, 0), 2, cv::LINE_8);
        cv::imshow("Histogram", hist_image);
        cv::waitKey(5);

        double red_line_length = cv::norm(intersection_point - pTh);
        std::cout << "triangleThreshold - Longitud de la línea roja perpendicular" << bin << ": " << red_line_length << std::endl;
        std::cout << "triangleThreshold - Umbral actual: " << threshold_val << std::endl;
        if (red_line_length > maxLength)
        {
            maxLength = red_line_length;
            threshold_val = bin;
        }
    }

    std::cout << "triangleThreshold - Umbral: " << threshold_val << std::endl;

    cv::Mat bin_img;
    cv::threshold(image, bin_img, threshold_val, 255, cv::THRESH_BINARY);

    return bin_img;
}

cv::Mat cumSum(const cv::Mat& hist)
{
    cv::Mat cumSumMat(hist.size(), hist.type());

    double sum = 0;
    for (int i = 0; i < hist.rows; ++i)
    {
        sum += hist.at<float>(i);
        cumSumMat.at<float>(i) = sum;
    }

    return cumSumMat;
}

int threshold_Yen(const cv::Mat& image)
{
    /* Ported to C++ by Alexander Ivakov from Java implementation of ImageJ plugin Auto_Threshold

        // Implements Yen  thresholding method
        // 1) Yen J.C., Chang F.J., and Chang S. (1995) "A New Criterion
        //    for Automatic Multilevel Thresholding" IEEE Trans. on Image
        //    Processing, 4(3): 370-378
        // 2) Sezgin M. and Sankur B. (2004) "Survey over Image Thresholding
        //    Techniques and Quantitative Performance Evaluation" Journal of
        //    Electronic Imaging, 13(1): 146-165
        //    http://citeseer.ist.psu.edu/sezgin04survey.html
        //
        // M. Emre Celebi
        // 06.15.2007
        // Ported to ImageJ plugin by G.Landini from E Celebi's fourier_0.8 routines
     */
    int threshold;
    int ih, it;
    double crit;
    double max_crit;
    double norm_histo[256]; /* normalized histogram */
    double P1[256]; /* cumulative normalized histogram */
    double P1_sq[256];
    double P2_sq[256];

    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&image, 1, nullptr, cv::noArray(), hist, 1, &histSize, &histRange);

    int total = 0;
    for (ih = 0; ih < 256; ih++)
        total += (int)hist.at<float>(ih);
    std::cout<< "Total: " << total << std::endl;
    for (int ih = 0; ih < 256; ih++)
        norm_histo[ih] = (double)hist.at<float>(ih) / total;

    P1[0] = norm_histo[0];
    for (ih = 1; ih < 256; ih++)
        P1[ih] = P1[ih-1] + norm_histo[ih];
    P1_sq[0] = norm_histo[0] * norm_histo[0];
    for (ih = 1; ih < 256; ih++)
        P1_sq[ih] = P1_sq[ih-1] + norm_histo[ih] * norm_histo[ih];
    P2_sq[255] = 0.0;
    for (ih = 254; ih >= 0; ih--)
        P2_sq[ih] = P2_sq[ih+1] + norm_histo[ih+1] * norm_histo[ih+1];

    /* Find the threshold that maximizes the criterion */
    threshold = -1;
    max_crit = std::numeric_limits<double>::min();
    for ( it = 0; it < 256; it++ )
    {
        crit = -1.0 * (( P1_sq[it] * P2_sq[it] )> 0.0?
                           std::log( P1_sq[it] * P2_sq[it]):0.0) +  2 * ( ( P1[it] * ( 1.0 - P1[it] ) )>0.0?
                          std::log(  P1[it] * ( 1.0 - P1[it] ) ): 0.0);
        if ( crit > max_crit )
        {
            max_crit = crit;
            threshold = it;
        }
    }
    return threshold;
}

/// @brief Método global pág. 7 Adaptive Segmentation Algorithm for Subtle Defect Images on the Surface of Magnetic Ring Using 2D-Gabor Filter Bank
///        Similar a Hampel pero con "k" empírico
int adaptiveZhangThreshold(cv::Mat& image)
{
    cv::Mat image_64F = dsi::MatrixFunctions::dsiConvertTo(image, CV_64FC1);

    cv::Scalar mean, stddev;
    cv::meanStdDev(image_64F, mean, stddev);

    // Ajustado experimentalmente en el paper
    double k = 1.0 + (1 - exp(0.60 - 0.0045 * mean.val[0])) / (1 + exp(0.60 - 0.0045 * mean.val[0]));

    double threshold = mean.val[0] - k * stddev.val[0];
    std::cout << "adaptiveZhangThreshold - threshold: " << threshold << std::endl;

    return std::min(std::max(0, (int)(threshold * 255.0)), 255);
}

cv::Mat threshold_minMax(const cv::Mat& src)
{
    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);
    double threshold = (minVal + maxVal) / 2.0;
    std::cout << "threshold_minMax - threshold: " << threshold << std::endl;
    cv::Mat dst;
    cv::threshold(src, dst, threshold, 255, cv::THRESH_BINARY);
    return dst;
}

cv::Mat threshold_grayLevelAverage(const cv::Mat& src)
{
    double average = cv::mean(src)[0];
    std::cout << "threshold_minMax - threshold: " << average << std::endl;
    cv::Mat dst;
    cv::threshold(src, dst, average, 255, cv::THRESH_BINARY);
    return dst;
}

cv::Mat threshold_histogramMinMax(const cv::Mat& src)
{
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&src, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange);

    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(hist, nullptr, nullptr, &minLoc, &maxLoc);

    double threshold = (minLoc.y + maxLoc.y) / 2.0;
    std::cout << "threshold_histogramMinMax - minLoc.y: " << minLoc.y << std::endl;
    std::cout << "threshold_histogramMinMax - maxLoc.y: " << maxLoc.y << std::endl;
    std::cout << "threshold_histogramMinMax - threshold: " << threshold << std::endl;

    cv::Mat dst;
    cv::threshold(src, dst, threshold, 255, cv::THRESH_BINARY);

    return dst;
}

/// @brief T_k_plus_one = ( ( sum_desde_0_hasta_Tk(value_hist * bin_hist) / (2 * sum_desde_0_hasta_Tk(value_hist)) )
///                       + ( sum_desde_Tk_hasta_N(value_hist * bin_hist) / (2 * sum_desde_Tk_hasta_N(value_hist)) ) )
/// @see https://www.researchgate.net/publication/3116609_Picture_Thresholding_Using_an_Iterative_Selection_Method_-_Comments
cv::Mat threshold_Trussell(const cv::Mat& src, int max_iters = 1000)
{
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&src, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange);
    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);
    double initial_threshold = (minVal + maxVal) / 2.0;

    int iter = 0;
    double old_threshold = 0.0, new_threshold = initial_threshold;
    while (true)
    {
        double num_below_threshold = 0.0;
        double den_below_threshold = 0.0;
        double num_above_threshold = 0.0;
        double den_above_threshold = 0.0;

        for (int i = 0; i < old_threshold; ++i)
        {
            num_below_threshold += i * hist.at<float>(i);
            den_below_threshold += hist.at<float>(i);
        }
        for (int i = new_threshold; i < histSize; ++i)
        {
            num_above_threshold += i * hist.at<float>(i);
            den_above_threshold += hist.at<float>(i);
        }

        den_below_threshold *= 2;
        den_above_threshold *= 2;

        new_threshold = (num_below_threshold / std::max(den_below_threshold, 1.0)) + (num_above_threshold / std::max(den_above_threshold, 1.0));

        std::cout << "threshold_Trussell - old_threshold: " << old_threshold << std::endl;
        std::cout << "threshold_Trussell - new_threshold: " << new_threshold << std::endl;

        if (std::abs(new_threshold - old_threshold) < 0.5 || iter >= max_iters)
            break;

        old_threshold = new_threshold;
        ++iter;
    }
    std::cout << "threshold_Trussell - threshold: " << (int)new_threshold << std::endl;

    // Dibujar histograma y valor umbral
    {
        cv::Mat hist_image(400, 512, CV_8UC3, cv::Scalar(255, 255, 255));
        int bin_width = hist_image.cols / histSize;
        cv::normalize(hist, hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());

        for (int i = 1; i < histSize; ++i)
        {
            cv::line(hist_image,
                     cv::Point(bin_width * (i - 1), hist_image.rows - cvRound(hist.at<float>(i - 1))),
                     cv::Point(bin_width * i, hist_image.rows - cvRound(hist.at<float>(i))),
                     cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
        }

        cv::line(hist_image,
                 cv::Point(bin_width * (int)new_threshold, hist_image.rows),
                 cv::Point(bin_width * (int)new_threshold, 0),
                 cv::Scalar(255, 0, 0), 2, cv::LINE_8);

        cv::imshow("Histogram", hist_image);
        cv::waitKey(0);
    }

    cv::Mat dst;
    cv::threshold(src, dst, (int)new_threshold, 255, cv::THRESH_BINARY);

    return dst;
}

int main()
{
    cv::Mat reconstructedImageMat = cv::imread("/home/alejandro/Escritorio/CIE_GALFOR/datosCompartidos/g.png", CV_LOAD_IMAGE_GRAYSCALE);
    if (reconstructedImageMat.empty())
    {
        std::cout << "Can't read Image. Try Different Format." << std::endl;
        exit(1);
    }
    cv::normalize(reconstructedImageMat, reconstructedImageMat, 0.0, 255.0, cv::NORM_MINMAX, CV_8UC1);
    reconstructedImageMat = dsi::MatrixFunctions::dsiConvertTo(reconstructedImageMat, CV_8UC1);
    muestraImagenOpenCV(reconstructedImageMat, "reconstructedImageMat", false);
    std::cout << "reconstructedImageMat.rows: " << reconstructedImageMat.rows << " reconstructedImageMat.cols: " << reconstructedImageMat.cols << std::endl;
    std::cout << "depth: " << reconstructedImageMat.depth() << " Channels: " << reconstructedImageMat.channels() << std::endl;

    /// Probar con slider umbral global
    {
        //        cv::namedWindow("Input Image", cv::WINDOW_NORMAL);
        //        cv::imshow("Input Image", reconstructedImageMat);
        //        cv::namedWindow("Binary Image", cv::WINDOW_NORMAL);
        //        int initialThreshold = 128; // Umbral inicial
        //        cv::createTrackbar("Threshold", "Binary Image", &initialThreshold, 255, onTrackbarChange, &reconstructedImageMat);
        //        onTrackbarChange(initialThreshold, &reconstructedImageMat);
        //        cv::waitKey(0);
    }

    //    std::vector<std::string> methods = {
    //        "Manual"
    //        "hist_minMax",
    //        "minMax"
    //        "grayLvlAvg",
    //        "Ridler",
    //        "Trussell",
    //        "Otsu",
    //        "Deriche",
    //        "Kittle",
    //        "Pun",
    //        "Kapur",
    //        "Johanssen",
    //        "Triangle"
    //        "Yen"
    //        "Zhang"
    //        "RenyiEntropy"
    //    };
    cv::Mat defectUmbralized(reconstructedImageMat.rows, reconstructedImageMat.cols, CV_8UC1);
    std::string method = "Zhang";
    double th = 0.0;
    if (method == "Otsu")
    {
        cv::threshold(reconstructedImageMat, defectUmbralized, 0.0, 255.0, CV_THRESH_BINARY | CV_THRESH_OTSU);
        std::cout << "Umbral: " << th << std::endl;
    }
    else if (method == "Manual")
    {
        th = 200.0;
        cv::threshold(reconstructedImageMat, defectUmbralized, th, 255.0, CV_THRESH_BINARY);
        std::cout << "Umbral: " << th << std::endl;
    }
    else if (method == "Triangle")
    {
        //        cv::threshold(reconstructedImageMat, defectUmbralized, th, 255.0, CV_THRESH_BINARY | CV_THRESH_TRIANGLE);
        defectUmbralized = triangleThreshold(reconstructedImageMat);
    }
    else if (method == "minMax")
    {
        defectUmbralized = threshold_minMax(reconstructedImageMat);
    }
    else if (method == "hist_minMax")
    {
        defectUmbralized = threshold_histogramMinMax(reconstructedImageMat);
    }
    else if (method == "grayLvlAvg")
    {
        defectUmbralized = threshold_grayLevelAverage(reconstructedImageMat);
    }
    else if (method == "Ridler")
    {

    }
    else if (method == "Trussell")
    {
        defectUmbralized = threshold_Trussell(reconstructedImageMat);
    }
    else if (method == "Deriche")
    {

    }
    else if (method == "Kittle")
    {

    }
    else if (method == "Pun")
    {

    }
    else if (method == "Kapur")
    {

    }
    else if (method == "Johanssen")
    {

    }
    else if (method == "RenyiEntropy")
    {

    }
    else if (method == "Yen")
    {
        th = threshold_Yen(reconstructedImageMat);
        cv::threshold(reconstructedImageMat, defectUmbralized, th, 255.0, CV_THRESH_BINARY);
        std::cout << "Umbral Yen: " << th << std::endl;
    }
    else if (method == "Zhang")
    {
        th = adaptiveZhangThreshold(reconstructedImageMat);
        cv::threshold(reconstructedImageMat, defectUmbralized, th, 255.0, CV_THRESH_BINARY);
        std::cout << "Umbral Zhang: " << th << std::endl;
    }
    muestraImagenOpenCV(defectUmbralized, "defectUmbralized", false);

    int morphoKernelSize = 5;
    cv::morphologyEx(defectUmbralized, defectUmbralized, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_RECT, cv::Size(morphoKernelSize, morphoKernelSize)));
    cv::morphologyEx(defectUmbralized, defectUmbralized, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_RECT, cv::Size(morphoKernelSize, morphoKernelSize)));
    cv::dilate(defectUmbralized, defectUmbralized, cv::Mat(), cv::Point(-1, -1), 3);
    muestraImagenOpenCV(defectUmbralized, "closed, opened and dilated defect map", false);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(defectUmbralized, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::cvtColor(reconstructedImageMat, reconstructedImageMat, cv::COLOR_GRAY2RGB);
    cv::drawContours(reconstructedImageMat, contours, -1, cv::Scalar(0, 0, 255), 2);
    muestraImagenOpenCV(reconstructedImageMat, "Detected defects", false);

    return 0;
}
