#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <chrono>

#include "rootsift.hpp"


#define IMG_DIR "~/data/reichstag/dense/images"


void
openCVSIFTDetect( const cv::Mat image )
{
    cv::SiftFeatureDetector sift;

    std::vector<cv::KeyPoint> keypoints;
    sift.detect(image, keypoints);

    cv::Mat descriptors;
    sift.compute(image, keypoints, descriptors);
}

void
meSIFTDetect( const cv::Mat image )
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    clock_t start = clock();
    meSIFT(image, cv::noArray(), keypoints, cv::noArray(), false);
    printf( "Time to run meSIFT for keypoints: %0.2f\n",
            ((float)clock() - start) / CLOCKS_PER_SEC );
    start = clock();
    meSIFT(image, cv::noArray(), keypoints, descriptors, true);
    printf( "Time to run meSIFT for descriptors: %0.2f\n",
            ((float)clock() - start) / CLOCKS_PER_SEC );
}

int
main( int argc, const char* argv[] )
{
    // load in greyscale input image
    const cv::Mat input_img = cv::imread(argv[1], 0);

    // cv::Mat desc1, desc2;
    clock_t start = clock();
    openCVSIFTDetect(input_img);
    printf( "Time to compute openCV SIFT: %0.2f\n",
            ((float)clock() - start) / CLOCKS_PER_SEC );

    clock_t start2 = clock();
    meSIFTDetect(input_img);
    printf( "Time to compute meSIFT: %0.2f\n",
            ((float)clock() - start2) / CLOCKS_PER_SEC );

    return 0;
}