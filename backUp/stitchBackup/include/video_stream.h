#ifndef VIDEO_STREAM_H
#define VIDEO_STREAM_H
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "opencv2/opencv.hpp"

namespace stitching {
    class VideoStream {
    public:
        VideoStream(std::string path, float compose_scale = 1., size_t queue_size = std::max(2u, static_cast<unsigned int>(std::thread::hardware_concurrency() * 0.5)));
        cv::VideoCapture getCapture();
        void stop();
        cv::Mat read();
        bool more();

    private:
        void update();

        const std::string path;
        const size_t queue_size;
        const float compose_scale;
        cv::VideoCapture capture;
        bool stopped = false;
        std::queue<cv::Mat> frames;
        std::mutex mutex;
        std::condition_variable cv;
        std::thread thread;
    };
}

#endif // VIDEO_STREAM_H
