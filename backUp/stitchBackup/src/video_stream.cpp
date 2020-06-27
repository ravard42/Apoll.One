#include "video_stream.h"

namespace stitching {
    VideoStream::VideoStream(std::string path, float compose_scale, size_t queue_size)
        : path(path)
        , queue_size(queue_size)
        , compose_scale(compose_scale)
        , capture(path) // TODO: figure it out: , cv::CAP_DSHOW)//"filesrc location=" + path + " ! avidemux ! vaapidecodebin  ! appsink", cv::CAP_GSTREAMER)
        , thread(std::thread(&VideoStream::update, this))
    {
    }

    cv::VideoCapture VideoStream::getCapture()
    {
        return capture;
    }

    void VideoStream::stop()
    {
        this->stopped = true;
        this->thread.join();
    }

    cv::Mat VideoStream::read()
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        cv.wait(lock, [this] { return this->more(); });
        cv::Mat frame = this->frames.front();
        this->frames.pop();
        return frame;
    }

    bool VideoStream::more()
    {
        return this->frames.size() > 0;
    }

    void VideoStream::update()
    {
        cv::Mat frame;
        while (!this->stopped && this->capture.isOpened()) {
            if (this->frames.size() < this->queue_size) {
                if (!this->capture.read(frame))
                    break;
                if (abs(compose_scale - 1) > 1e-1)
                    cv::resize(frame, frame, cv::Size(), compose_scale, compose_scale, cv::INTER_LINEAR);
                std::unique_lock<std::mutex> lock(this->mutex);
                this->frames.push(frame);
                lock.unlock();
                cv.notify_one();
            } else
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        this->stopped = true;
    }
}
