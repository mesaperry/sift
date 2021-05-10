#include <opencv2/nonfree/features2d.hpp>


struct thread_data
{
    const std::vector<cv::Mat>& gpyr;
    const std::vector<cv::KeyPoint>& keypoints;
    cv::Mat& descriptors;
    int nOctaveLayers;
    int firstOctave;
    unsigned long thread_id;
    thread_data( const std::vector<cv::Mat>& gpyr,
                 const std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors )
                   : gpyr(gpyr), keypoints(keypoints), descriptors(descriptors) {}
};

void meSIFT( cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&,
             cv::OutputArray, bool );
