#ifndef UTILS_H
#define UTILS_H

#include "opencv2/core/utility.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/detail/matchers.hpp"

namespace cvd = cv::detail;

namespace stitching {

    struct StitchingInfos {
        StitchingInfos() = default;
        StitchingInfos(const std::vector<cvd::CameraParams>& cameras, double warped_image_scale, double work_scale, double score, std::vector<cvd::ImageFeatures> features, std::vector<cvd::MatchesInfo> pairwise_matches);

        std::vector<cvd::CameraParams> cameras;
        double warped_image_scale = 1.0;
        double work_scale = 1.;
        double score = 0.0;
		std::vector<cvd::ImageFeatures> features;
		std::vector<cvd::MatchesInfo> pairwise_matches;
    };

    struct ComposeInfos {
        ComposeInfos(const std::vector<cv::Mat>& mask_warped, const std::vector<cv::Point>& offsets, cv::Size pano_frame_size, const std::vector<cv::Size>& warped_sizes, float compose_scale, const std::vector<cv::Mat>& cameras_r, const std::vector<cv::Mat>& cameras_k, const std::vector<cv::Point>& corners);
		void Save(std::string filename = "compose_infos.yml") const;
		static cv::Ptr<ComposeInfos> Load(std::string filename = "compose_infos.yml");

        const std::vector<cv::Mat> mask_warped, cameras_r, cameras_k;
        const cv::Size pano_frame_size;
		const std::vector<cv::Size> warped_sizes;
		const float compose_scale;
		const std::vector<cv::Point> offsets;
		const std::vector<cv::Point> corners;
    };

    // Shows given image in a window, usefull for debugging purpose
    void show_img(cv::InputArray img, std::string title = "image");
	// Resizes two frames given a float scale
    void scale_frames(cv::Mat& frame1, cv::Mat& frame2, float compose_scale);
}

#endif // UTILS_H