#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <chrono>

#include "rootsift.hpp"


#define IMG_DIR "~/data/reichstag/dense/images"


void
openCVSIFTDetect( const cv::Mat image )
{
    cv::SiftFeatureDetector detector;
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(image, keypoints);

    // render and output image
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);
    cv::imwrite("opencv_res.jpg", output);
}

void
meSIFTDetect( const cv::Mat image )
{
    std::vector<cv::KeyPoint> keypoints;
    meSIFT(image, cv::noArray(), keypoints, cv::noArray(), false);

    // render and output image
    cv::Mat output;
    cv::drawKeypoints(image, keypoints, output);
    cv::imwrite("me_res.jpg", output);
}


int
main( int argc, const char* argv[] )
{
    // load in greyscale input image
    const cv::Mat input_img = cv::imread(argv[1], 0);

    clock_t start = clock();
    openCVSIFTDetect(input_img);
    printf( "Time to compute openCV SIFT: %0.2f\n",
            ((float)clock() - start) / CLOCKS_PER_SEC );

    start = clock();
    meSIFTDetect(input_img);
    printf( "Time to compute meSIFT: %0.2f\n",
            ((float)clock() - start) / CLOCKS_PER_SEC );

    return 0;
}