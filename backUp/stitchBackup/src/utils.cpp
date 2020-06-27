#include "opencv2/highgui.hpp"

#include "utils.h"

namespace stitching {
    StitchingInfos::StitchingInfos(const std::vector<cvd::CameraParams>& cameras, double warped_image_scale, double work_scale, double score, std::vector<cvd::ImageFeatures> features, std::vector<cvd::MatchesInfo> pairwise_matches)
        : cameras(cameras)
        , warped_image_scale(warped_image_scale)
        , work_scale(work_scale)
        , score(score)
		, features(features)
		, pairwise_matches(pairwise_matches)
    {
    }

	ComposeInfos::ComposeInfos(const std::vector<cv::Mat>& mask_warped, const std::vector<cv::Point>& offsets, cv::Size pano_frame_size, const std::vector<cv::Size>& warped_sizes, float compose_scale, const std::vector<cv::Mat>& cameras_r, const std::vector<cv::Mat>& cameras_k, const std::vector<cv::Point>& corners)
		: mask_warped(mask_warped)
		, pano_frame_size(pano_frame_size)
		, warped_sizes(warped_sizes)
		, compose_scale(compose_scale)
		, cameras_r(cameras_r)
		, cameras_k(cameras_k)
		, offsets(offsets)
		, corners(corners)
	{
		const size_t common_size = mask_warped.size();
		assert(offsets.size() == common_size && warped_sizes.size() == common_size && cameras_r.size() == common_size && cameras_k.size() == common_size);
	}
	
	cv::Ptr<ComposeInfos> ComposeInfos::Load(std::string filename)
	{
		cv::FileStorage file(filename, cv::FileStorage::READ);
		int common_size = static_cast<int>(file["common_size"]);

		auto deserialize_vector = [&file, common_size](std::string prefix, auto& vector) {
			vector.resize(common_size);
			for (size_t idx = 0; idx < vector.size(); ++idx)
				file[prefix + "_" + std::to_string(idx)] >> vector[idx];
		};

		std::vector<cv::Mat> masks;
		deserialize_vector("masks", masks);

		std::vector<cv::Point> offsets;
		deserialize_vector("offsets", offsets);

		cv::Size pano_frame_size;
		file["pano_frame_size"] >> pano_frame_size;

		std::vector<cv::Size> warped_sizes;
		deserialize_vector("warped_sizes", warped_sizes);

		std::vector<cv::Point> corners;
		deserialize_vector("corners", corners);

		float compose_scale = static_cast<float>(file["compose_scale"]);

		std::vector<cv::Mat> cameras_r;
		deserialize_vector("R", cameras_r);

		std::vector<cv::Mat> cameras_k;
		deserialize_vector("K", cameras_k);

		file.release();

		return cv::makePtr<ComposeInfos>(masks, offsets, pano_frame_size, warped_sizes, compose_scale, cameras_r, cameras_k, corners);
	}

	void ComposeInfos::Save(std::string filename) const
	{
        cv::FileStorage file(filename, cv::FileStorage::WRITE);
		file << "common_size" << static_cast<int>(mask_warped.size());

		auto serialize_vector = [&file](std::string prefix, const auto& vector) {
			for (size_t idx = 0; idx < vector.size(); ++idx)
				file << prefix + '_' + std::to_string(idx) << vector[idx];
		};

		serialize_vector("warped_sizes", warped_sizes);
		serialize_vector("offsets", offsets);
		serialize_vector("corners", corners);
		serialize_vector("R", cameras_r);
		serialize_vector("K", cameras_k);

		file << "pano_frame_size" << pano_frame_size;
		file << "compose_scale" << compose_scale;
		serialize_vector("masks", mask_warped);
		file.release();
	}

    void show_img(cv::InputArray img, std::string title)
    {
        cv::imshow(title, img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }


    void scale_frames(cv::Mat& frame1, cv::Mat& frame2, float compose_scale)
    {
        if (std::abs(compose_scale - 1) > 1e-1) {
            cv::resize(frame1, frame1, cv::Size(), compose_scale, compose_scale, cv::INTER_LINEAR);
            cv::resize(frame2, frame2, cv::Size(), compose_scale, compose_scale, cv::INTER_LINEAR);
        }
    }
}
