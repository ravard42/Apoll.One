#define BLENDSTRENGHT 50
//#define CSS1 6.0  23/12/18 in param now see runMulti.sh
#define CSS2 12.0 //(center_seam_size = 1 / CSS2)
//#define MATCH_CONF 0.70 25/12/18 in param

#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <time.h>
#include <cstdio>
#include <utility>
#include <signal.h>
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/detail/blenders.hpp"


#include "utils.h"
#include "argparse.h" // Code from https://github.com/hbristow/argparse
#include "video_stream.h"

namespace stitching {
    // This is a modified version of OpenCV built-in stitching pipeline (see https://docs.opencv.org/2.4/modules/stitching/doc/introduction.html and https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp)
    // This pipeline implements stitching for panorama video generation from two cameras.
    //TODO: set higher priority to worker threads
    //TODO: adapt video stream queue size and thread number to available RAM and video memory
    //TODO: add options to flip video and syncrhonize files with audio?
    //TODO: use nvvl instead of videoCaptures to decrypt videos faster (see https://github.com/NVIDIA/nvvl) (frames can be loaded directly into a gpuMat, GPU implementation of sphericalWarper may be worth it with nvvl)
    //TODO: (re)encode videos using VP9 (royalty free equivalent to H265) instead of H264

	//nom de l'executable compilé que l'on utilisera comme nom de dossier relatif
	std::string dirName;

    // Default command line args
	const int blend_type = cvd::Blender::MULTI_BAND; // Blending method (MULTI_BAND, FEATHER or NO)
	const float blend_strength = BLENDSTRENGHT; // Blending strength from [0,100] range.

    const bool orb_feature_finder = false; // Use ORB feature finder (surf otherwise)
    const int range_width = 2; // limit number of images to match with.
    //float match_conf = 0.5f; // Confidence for feature matching step (default must be around 0.3 for orb feature finder and 0.65)
    const float conf_thresh = 0.; //1.1f;		// Threshold for two images are from the same panorama confidence
	const double work_megapix = -1; // Resolution for image registration step
	const double compose_megapix = -1; // Resolution for image registration step
    const double seam_megapix = 1.0; // Resolution for seam estimation step
    const bool do_wave_correct = true; // Enable wave effect correction
	const std::string ba_refine_mask = "_____"; // Set refinement mask for bundle adjustment.  'x' means refine respective parameter and '_' means don't
	//const std::string ba_refine_mask = "x__x_"; // Set refinement mask for bundle adjustment.  'x' means refine respective parameter and '_' means don't
    const bool draw_matches = true; // Draw features matches for debugging purpose
	const bool measure_stitching_perfs = true;
    const size_t estimation_trials = 60;
	const double alphaOptimMatrix = 0.8;
	const int seam_reprocess_cycle = 20;
    float resize_factor = 0.5; // May adapt to input resolution to avoid too large panoramas
    //float resize_factor = 0.5; // May adapt to input resolution to avoid too large panoramas
	StitchingInfos stitching_infos;
	cv::VideoWriter panoramaVideo;
	// Color correction parameters
	double color_correction_img_idx = 0;
	double color_correction_L_fact = 1;
	double color_correction_L_shift = 0;
	double color_correction_a_fact = 1;
	double color_correction_a_shift = 0;
	double color_correction_b_fact = 1;
	double color_correction_b_shift = 0;


	// variables initialisées dans runMulti.sh
	int total_shift, left_shift, right_shift;
	int haut, bas;
	bool rectify_map;
   float match_conf; // Confidence for feature matching step (default must be around 0.3 for orb feature finder and 0.65)
	float css1;
	int offset_verifPano;
	//

	//const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 857.0, 0, 968.0, 0, 1280.0, 720.0, 0, 0, 1.0);
	//const cv::Mat cameraMatrixR = (cv::Mat_<double>(3, 3) << resize_factor * 1716.0, 0, resize_factor * 1716.0, 0, resize_factor * 800.0, resize_factor * 720.0, 0, 0, 1.0);
	//const cv::Mat cameraMatrixL = (cv::Mat_<double>(3, 3) << resize_factor * 1716.0, 0, resize_factor * 1716.0, 0, resize_factor * 800.0, resize_factor * 720.0, 0, 0, 1.0);

	const cv::Mat cameraMatrixR = (cv::Mat_<double>(3, 3) << resize_factor * 1.33075325e+03, resize_factor * 0.00000000e+00, resize_factor * 1.25153018e+03, 0, resize_factor * 1.31851764e+03, resize_factor * 7.30357002e+02, 0, 0, 1.0);
	const cv::Mat cameraMatrixL = (cv::Mat_<double>(3, 3) << resize_factor * 1.33075325e+03, resize_factor * 0.00000000e+00, resize_factor * 1.25153018e+03, 0, resize_factor * 1.31851764e+03, resize_factor * 7.30357002e+02, 0, 0, 1.0);

	//const cv::Mat distCoeffsL = (cv::Mat_<double>(5, 1) << -0., 0.0, 0, 0, 0);
	//const cv::Mat distCoeffsR = (cv::Mat_<double>(5, 1) << -0., 0.0, 0, 0.0, 0);

	const cv::Mat distCoeffsL = (cv::Mat_<double>(5, 1) << -0.33670671, 0.10413804, -0.00164152, -0.00052342, 0.);
	const cv::Mat distCoeffsR = (cv::Mat_<double>(5, 1) << -0.33670671,  0.10413804, - 0.00164152, - 0.00052342,  0.);
	const cv::Mat distCoeffsL_f = (cv::Mat_<double>(5, 1) << -0.33670671, 0.10413804, 0., 0., 0.);
	const cv::Mat distCoeffsR_f = (cv::Mat_<double>(5, 1) << -0.33670671, 0.10413804, 0., 0., 0.);

    StitchingInfos find_stitching_infos(const std::vector<cv::Mat>& full_images, StitchingInfos& prev_stitching_infos);
	cv::Ptr<ComposeInfos> find_seam_masks(const std::vector<cv::Mat>& full_images, const StitchingInfos& stitching_infos);
	cv::Ptr<ComposeInfos> find_seam_masks(const std::vector<cv::Mat>& full_images, const StitchingInfos& stitching_infos, std::vector<cv::Mat> prev_seamed_mask);
	

    static double get_scale(const std::vector<cv::Mat>& full_images, double megapix)
    {
        return (megapix < 0 || full_images.size() == 0) ? 1.0 : std::min(1.0, sqrt(megapix * 1e6 / full_images[0].size().area()));
    }
	void equalize(cv::UMat& frame, cv::Mat& dst)
	{
		cv::Mat lab_image;
		cv::cvtColor(frame, lab_image, cv::COLOR_RGB2Lab);

		// Extract the L channel
		std::vector<cv::Mat> lab_planes(3);
		cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

										   // apply the CLAHE algorithm to the L channel
		cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
		clahe->setClipLimit(2);
		clahe->setTilesGridSize(cv::Size(8, 8));
		cv::Mat tmp;
		clahe->apply(lab_planes[0], tmp);

		// Merge the the color planes back into an Lab image
		tmp.copyTo(lab_planes[0]);
		cv::merge(lab_planes, lab_image);

		// convert back to RGB
		cv::cvtColor(lab_image, dst, cv::COLOR_Lab2LRGB);
	}
	void color_correction_estimator(cv::Mat frame1, cv::Mat frame2, cv::Mat& output_frame)
	{
		/* Now convert them into CIE Lab color space */
		cv::Mat source_img_cie,
			target_img_cie;

		cv::cvtColor(frame1, source_img_cie, cv::COLOR_BGR2Lab);
		cv::cvtColor(frame2, target_img_cie, cv::COLOR_BGR2Lab);


		/* Split into individual l a b channels */
		std::vector<cv::Mat> source_channels,
			target_channels;

		cv::split(source_img_cie, source_channels);
		cv::split(target_img_cie, target_channels);

		/* For each of the l, a, b, channel ... */
		for (int i = 0; i < 3; i++) {
			/* ... find the mean and standard deviations */
			/* ... for source image ... */
			cv::Mat temp_mean, temp_stddev;
			meanStdDev(source_channels[i], temp_mean, temp_stddev);
			double source_mean = temp_mean.at<double>(0);
			double source_stddev = temp_stddev.at<double>(0);

			/* ... and for target image */
			meanStdDev(target_channels[i], temp_mean, temp_stddev);
			double target_mean = temp_mean.at<double>(0);
			double target_stddev = temp_stddev.at<double>(0);

			/* Fit the color distribution from target LAB to our source LAB */
			target_channels[i].convertTo(target_channels[i], CV_64FC1);
			target_channels[i] -= target_mean;
			target_channels[i] *= (target_stddev / source_stddev);
			target_channels[i] += source_mean;
			target_channels[i].convertTo(target_channels[i], CV_8UC1);
		}


		/* Merge the lab channels back into a single BGR image */
		cv::Mat output_img;
		cv::merge(target_channels, output_img);
		cv::cvtColor(output_img, output_frame, cv::COLOR_Lab2BGR);
	}
	cv::Ptr<cvd::Blender> prepareBlender(const ComposeInfos& compose_infos)
	{
		static thread_local cv::Ptr<cvd::Blender> blender;

		if (!blender) {
			blender = cvd::Blender::createDefault(blend_type, true);
			if (blend_type != cvd::Blender::NO) {
				cv::Size dst_sz = cvd::resultRoi(compose_infos.corners, compose_infos.warped_sizes).size();
				float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
				if (blend_width < 1.f)
					blender = cvd::Blender::createDefault(cvd::Blender::NO, true);
				else if (blend_type == cvd::Blender::MULTI_BAND) {
					cvd::MultiBandBlender* mb = dynamic_cast<cvd::MultiBandBlender*>(blender.get());
					mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
				}
				else if (blend_type == cvd::Blender::FEATHER) {
					cvd::FeatherBlender* fb = dynamic_cast<cvd::FeatherBlender*>(blender.get());
					fb->setSharpness(1.f / blend_width);
				}
			}
		}

		blender->prepare(compose_infos.corners, compose_infos.warped_sizes);
		return blender;
	}

    cv::Ptr<cvd::RotationWarper> create_warper(double warped_image_scale)
    {
        cv::Ptr<cv::WarperCreator> warper_creator;

        /* TODO: make sure SphericalWarperGpu isn't faster (especially if frames are decoded directly into a gpuMat)
		#ifdef HAVE_OPENCV_CUDAWARPING
		if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
		warper_creator = cv::makePtr<cv::SphericalWarperGpu>();
		} else
		#endif
		{*/
		//warper_creator = cv::makePtr<cv::TransverseMercatorWarper>();
		warper_creator = cv::makePtr<cv::SphericalWarper>();
		//warper_creator = cv::makePtr<cv::CylindricalWarper>();
        return warper_creator->create(static_cast<float>(warped_image_scale));
    }

	cv::Ptr<ComposeInfos> find_compose_infos(std::string video1, std::string video2) {
		cv::VideoCapture cap1(video1);
		cv::VideoCapture cap2(video2);
		if (!cap1.isOpened() || !cap2.isOpened()) {
			std::cout << "Can't open given video files." << std::endl;
			return cv::Ptr<ComposeInfos>();
		}

		// We assume both videos have the same resolution, framerate and codec
		size_t frame_count = static_cast<size_t>(std::min(cap1.get(cv::CAP_PROP_FRAME_COUNT)-left_shift, cap2.get(cv::CAP_PROP_FRAME_COUNT)-right_shift)-total_shift);
		float megapix = static_cast<float>(cap2.get(cv::CAP_PROP_FRAME_WIDTH) * cap2.get(cv::CAP_PROP_FRAME_HEIGHT));
		if (megapix > 6000000)
			resize_factor = std::min(resize_factor, 6000000.f / megapix); // Adapts to input resolution to avoid too large panoramas
		StitchingInfos best_infos;
		cv::Ptr<ComposeInfos> compose_infos;

		double frame_id = 0;

		std::default_random_engine generator(static_cast<unsigned int>(time(0)));
		std::uniform_int_distribution<> frame_distr(int(0), int(frame_count * 0.9));
	
		for (size_t i = 0; i < estimation_trials;) {
			//std::cout << "externe X" << std::endl;

			
			frame_id = static_cast<double>(frame_distr(generator));
			
			cap1.set(cv::CAP_PROP_POS_FRAMES, frame_id + left_shift + total_shift);
			cap2.set(cv::CAP_PROP_POS_FRAMES, frame_id + right_shift + total_shift);
			
			std::cout << "Finding stitching informations from frame#" << frame_id << " idx " << i << std::endl;
			cv::Mat frame1, frame2, frame_1, frame_2, map1L, map2L, map1R, map2R;
			//std::cout << "externe 0" << std::endl;

			try {
				if (!cap1.read(frame_1) || !cap2.read(frame_2))
					throw std::runtime_error("Can't read frame from video capture.");

				//scale_frames(frame_1, frame_2, resize_factor);
				cv::flip(frame_1, frame_1, -1);
				//cv::flip(frame_2, frame_2, -1);
				if (rectify_map) {
					scale_frames(frame_1, frame_2, resize_factor);
					cv::initUndistortRectifyMap(
						cameraMatrixL, distCoeffsL, cv::Mat(),
						getOptimalNewCameraMatrix(cameraMatrixL, distCoeffsL_f, frame_1.size(), alphaOptimMatrix, frame_1.size(), 0, true), frame_1.size(),
						CV_16SC2, map1L, map2L);
					remap(frame_1, frame1, map1L, map2L, cv::INTER_LINEAR);
					cv::initUndistortRectifyMap(
						cameraMatrixR, distCoeffsR, cv::Mat(),
						getOptimalNewCameraMatrix(cameraMatrixR, distCoeffsR_f, frame_1.size(), alphaOptimMatrix, frame_1.size(), 0, true), frame_1.size(),
						CV_16SC2, map1R, map2R);
					//cv::flip(frame_2, frame_2, 1);
					remap(frame_2, frame2, map1R, map2R, cv::INTER_LINEAR);
					//cv::flip(frame_2, frame_2, 1);
				} else {
					frame1 = frame_1;
					frame2 = frame_2;
					scale_frames(frame1, frame2, resize_factor);
				}

				//cv::flip(frame2, frame2, 1);
				//cv::initUndistortRectifyMap(
				//	cameraMatrixL, distCoeffsL, cv::Mat(),
				//	getDefaultNewCameraMatrix(cameraMatrixL, frame_1.size()), frame_1.size(),
				//	CV_16SC2, map1L, map2L);
				//remap(frame_1, frame1, map1L, map2L, cv::INTER_LINEAR);
				//cv::initUndistortRectifyMap(
				//	cameraMatrixR, distCoeffsR, cv::Mat(),
				//	getDefaultNewCameraMatrix(cameraMatrixR, frame_1.size()), frame_1.size(),
				//	CV_16SC2, map1R, map2R);
				//remap(frame_2, frame2, map1R, map2R, cv::INTER_LINEAR);


				try {
					//std::cout << "externe 1" << std::endl;

					StitchingInfos infos = find_stitching_infos({ frame1, frame2 }, stitching_infos);
					//std::cout << "externe 3" << std::endl;

					std::cout << "\tstitching infos score: " << infos.score << std::endl;
					if (infos.score > best_infos.score) {
						best_infos = infos;
						stitching_infos = best_infos;
					}
				}
				catch (const std::exception& e) {
					std::cout << "error" << e.what() << std::endl;
				}
				i++; // We increment i here so that we don't count exceptions as trials
			}
			catch (const std::runtime_error& err) {
				std::cout << err.what() << std::endl;
			}
			// We use a score limit to stabilize further processing.
			if (best_infos.score > 140 || best_infos.pairwise_matches.size() > 40) {
				break;
			}
			// We decrease match_conf if we don't detect any features so that the algorithm
			// adapt to luminosity, blurr effects.
			if (best_infos.score == 0 && i == 5) {
				std::cout << "up1" << std::endl;
				match_conf -= 0.2;
				i++;
			}
			if (best_infos.score == 0 && i == 15) {
				std::cout << "up2" << std::endl;
				match_conf -= 0.2;
				i++;
			}
			if (best_infos.score == 0 && i == 45) {
				std::cout << "up3" << std::endl;
				match_conf -= 0.8;
				i++;
			}
		}
		std::cout << "Best feature matching score found: " << best_infos.score << std::endl;

		std::cout << "Find masks from best camera infos found" << std::endl;
		cap1.set(cv::CAP_PROP_POS_FRAMES, frame_id + left_shift + total_shift);
		// Correction fps double a droite
		//cap2.set(cv::CAP_PROP_POS_FRAMES, 2 * (frame_id + right_shift + total_shift));
		cap2.set(cv::CAP_PROP_POS_FRAMES, frame_id + right_shift + total_shift);
		cv::Mat frame1, frame2, frame_1, frame_2, map1L, map2L, map1R, map2R;
		if (!cap1.read(frame_1) || !cap2.read(frame_2)) {
			std::cout << "Can't read frames from videos." << std::endl;
			return cv::Ptr<ComposeInfos>();
		}
		cap1.release();
		cap2.release();


		cv::flip(frame_1, frame_1, -1);
		//cv::flip(frame_2, frame_2, -1);
		if (rectify_map) {
			scale_frames(frame_1, frame_2, resize_factor);
			cv::initUndistortRectifyMap(
				cameraMatrixL, distCoeffsL, cv::Mat(),
				getOptimalNewCameraMatrix(cameraMatrixL, distCoeffsL_f, frame_1.size(), alphaOptimMatrix, frame_1.size(), 0, true), frame_1.size(),
				CV_16SC2, map1L, map2L);
			remap(frame_1, frame1, map1L, map2L, cv::INTER_LINEAR);
			cv::initUndistortRectifyMap(
				cameraMatrixR, distCoeffsR, cv::Mat(),
				getOptimalNewCameraMatrix(cameraMatrixR, distCoeffsR_f, frame_1.size(), alphaOptimMatrix, frame_1.size(), 0, true), frame_1.size(),
				CV_16SC2, map1R, map2R);
			//cv::flip(frame_2, frame_2, 1);
			remap(frame_2, frame2, map1R, map2R, cv::INTER_LINEAR);
			//cv::flip(frame2, frame2, 1);
		} else {
			frame1 = frame_1;
			frame2 = frame_2;
			scale_frames(frame1, frame2, resize_factor);
		}
		
		compose_infos = find_seam_masks({ frame1, frame2 }, best_infos);
		(*compose_infos).Save("Compose.yml");
		return compose_infos;
	}

	// Keyboard Interrupt Handler
	void consoleHandler(int signal) {

		printf("Ctrl-C handled\n"); // do cleanup
		panoramaVideo.release();
		exit(1);
	}

    int stitch_video(std::string video1, std::string video2, const ComposeInfos& compose_infos, std::string outputPanorama)
    {
        VideoStream stream1(video1, resize_factor);
        VideoStream stream2(video2, resize_factor);
		auto cap1 = stream1.getCapture(), cap2 = stream2.getCapture();

		if (!cap1.isOpened() || !cap2.isOpened()) {
			std::cout << "Can't open given video files (for panorama creation)." << std::endl;
			return EXIT_FAILURE;
		}

		for (int z =0; z<total_shift; z++) {
			stream1.read();
			stream2.read();
		}
		for (int z =0; z<left_shift; z++) {
			stream1.read();
		}
		for (int z =0; z<right_shift; z++) {
			stream2.read();
		}
		std::cout << "FPS : " << cap1.get(cv::CAP_PROP_FPS) << std::endl;
		std::cout << "FPS2 : " << cap2.get(cv::CAP_PROP_FPS) << std::endl;
		size_t frame_count = static_cast<size_t>(std::min(cap1.get(cv::CAP_PROP_FRAME_COUNT)-left_shift, cap2.get(cv::CAP_PROP_FRAME_COUNT)-right_shift)-total_shift);
		const double fps = cap2.get(cv::CAP_PROP_FPS);
		const int codec = static_cast<int>(cap1.get(cv::CAP_PROP_FOURCC));
		//std::cout << "Applying stitching informations to the whole video." << compose_infos.pano_frame_size << std::endl;
		const cv::Size pano_frame_size = compose_infos.pano_frame_size;
		panoramaVideo.open(outputPanorama, codec, fps, pano_frame_size, true);
		if (!panoramaVideo.isOpened()) {
			std::cout << "Can't create video writer for panorama output." << std::endl;
			return EXIT_FAILURE;
		}

		  //frame_count = 50000	; // TODO: remove it
        std::queue<cv::Mat> pano_frames;
        std::mutex writer_mutex;
        std::condition_variable frame_available;

        std::thread writer_thread([&] {
            for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
                std::unique_lock<std::mutex> lock(writer_mutex);
                frame_available.wait(lock, [&pano_frames] { return !pano_frames.empty(); });
                panoramaVideo.write(pano_frames.front());
                pano_frames.pop();
            }
        });

		cv::Ptr<cvd::RotationWarper> warper;
		cv::UMat temp_frame1, temp_frame2, warped_img1, warped_img2;
		cv::UMat mask = compose_infos.mask_warped[0].getUMat(cv::ACCESS_READ);
		//std::cout << "ATTENTION WARPER SCALE 2 " << stitching_infos.warped_image_scale << std::endl;
		warper = create_warper(stitching_infos.warped_image_scale);
		temp_frame1.create(pano_frame_size, CV_8UC3);
		temp_frame2.create(pano_frame_size, CV_8UC3);

        auto e1 = cv::getTickCount();

		// Keyboard Interrupt Handler
		struct sigaction sigIntHandler;
		sigIntHandler.sa_handler = consoleHandler;
		sigemptyset(&sigIntHandler.sa_mask);
		sigIntHandler.sa_flags = 0;
		sigaction(SIGINT, &sigIntHandler, NULL);

		std::vector<cv::Mat> prev_seam_mask = std::vector<cv::Mat>();
		cv::Ptr<ComposeInfos> currentComposePtr;
		// Stitching loop for each frame
		
		  int	verifId = 0;
        for (size_t frame_idx = 0; frame_idx < frame_count; ++frame_idx) {
			cv::Ptr<cvd::Blender> blender = prepareBlender(compose_infos);
			if (measure_stitching_perfs && frame_idx % 10 == 0)
				std::cout << "Processing frames " << frame_idx << " to " << frame_idx + 10 << " over " << frame_count << std::endl;
			auto offset1 = compose_infos.offsets[0], offset2 = compose_infos.offsets[1];

			cv::Mat frame_1, frame_2, frame_1_c, frame_2_c, map1L, map2L, map1R, map2R;
			//cv::flip(stream1.read(), frame_1, -1);
			frame_1 = stream1.read();
			frame_2 = stream2.read();
			
			cv::flip(frame_1, frame_1, -1);
			//cv::flip(frame_2, frame_2, -1);
			////scale_frames(frame_1, frame_2, resize_factor);


			if(rectify_map) {
				cv::initUndistortRectifyMap(
					cameraMatrixL, distCoeffsL, cv::Mat(),
					getOptimalNewCameraMatrix(cameraMatrixL, distCoeffsL_f, frame_1.size(), alphaOptimMatrix, frame_1.size(), 0, true), frame_1.size(),
					CV_16SC2, map1L, map2L);
				remap(frame_1, frame_1_c, map1L, map2L, cv::INTER_LINEAR);
				cv::initUndistortRectifyMap(
					cameraMatrixR, distCoeffsR, cv::Mat(),
					getOptimalNewCameraMatrix(cameraMatrixR, distCoeffsR_f, frame_1.size(), alphaOptimMatrix, frame_1.size(), 0, true), frame_1.size(),
					CV_16SC2, map1R, map2R);
				//cv::flip(frame_2, frame_2, 1);
				remap(frame_2, frame_2_c, map1R, map2R, cv::INTER_LINEAR);
				//cv::flip(frame_2_c, frame_2_c, 1);
			} else {
				//scale_frames(frame_1, frame_2, resize_factor);
				frame_1_c = frame_1;
				frame_2_c = frame_2;
			}

			//std::cout << "Marker1" << frame_1_c.size() << std::endl;
			//std::cout << "Marker1" << frame_2_c.size() << std::endl;
			//scale_frames(frame_1_c, frame_2_c, resize_factor);
			//cv::Ptr<ComposeInfos> infos = find_seam_masks({ frame_1_c, frame_2_c }, stitching_infos);
			//ComposeInfos compose_infos2 = *infos;
			std::vector<cv::Mat> images;
			images.push_back(frame_1_c);
			images.push_back(frame_2_c);
			if (frame_idx % seam_reprocess_cycle == 0) {
				currentComposePtr = find_seam_masks(images, stitching_infos, prev_seam_mask);
				prev_seam_mask = currentComposePtr->mask_warped;
			}
			ComposeInfos currentCompose = *currentComposePtr;


			warper->warp(
				frame_1_c, 
				currentCompose.cameras_k[0], 
				currentCompose.cameras_r[0],
				cv::INTER_LINEAR, 
				cv::BORDER_CONSTANT, 
				warped_img1
			);
			warper->warp(
				frame_2_c,
				currentCompose.cameras_k[1],
				currentCompose.cameras_r[1],
				cv::INTER_LINEAR, 
				cv::BORDER_CONSTANT, 
				warped_img2
			);
			//std::cout << "Marker2" << compose_infos.corners[0] << std::endl;
			//std::cout << "Marker2" << compose_infos.corners[1] << std::endl;
			//std::cout << "Marker2" << warped_img1.getMat(cv::ACCESS_READ).at<float>(64,89) << std::endl;
			//std::cout << "Marker2" << warped_img2.getMat(cv::ACCESS_READ).at<float>(64, 89) << std::endl;
			//const std::string filename = "oktest" + std::to_string(frame_idx) + ".jpg";
			//cv::imwrite(filename, warped_img2.getMat(cv::ACCESS_READ));
			
			// GAMMA CORRECTION
			cv::Mat frame1, frame2;
			//equalize(warped_img1, frame1);
			//equalize(warped_img2, frame2);
			
//			color_correction_estimator(warped_img2.getMat(cv::ACCESS_READ), warped_img1.getMat(cv::ACCESS_READ), frame1);
//			warped_img1 = frame1.getUMat(cv::ACCESS_READ);

			cv::Mat img_warped_s1, img_warped_s2;
			warped_img1.convertTo(img_warped_s1, CV_16S);
			warped_img2.convertTo(img_warped_s2, CV_16S);

			//std::cout << "Normal size : " << warped_img1.size() << "channels : " << warped_img1.channels() << "type : " << warped_img1.type() << std::endl;
			//std::cout << "Split size : " << frame1.size() << "channels : " << frame1.channels() << "type : " << frame1.type() << std::endl;
			//blender->feed(frame1, currentCompose.mask_warped[0], currentCompose.corners[0]);
			blender->feed(warped_img1, currentCompose.mask_warped[0], currentCompose.corners[0]);
			//blender->feed(warped_img1, compose_infos.mask_warped[0], compose_infos.corners[0]);
			//std::cout << "before feeding blender2" << std::endl;

			//blender->feed(frame2, currentCompose.mask_warped[1], currentCompose.corners[1]);
			blender->feed(warped_img2, currentCompose.mask_warped[1], currentCompose.corners[1]);
			//std::cout << "\tstitching gamma correction5 : " << temp_gamma_corrector << std::endl;
			//blender->feed(warped_img2, compose_infos.mask_warped[1], compose_infos.corners[1]);
			//std::cout << "before blending" << std::endl;
			//warped_img1.release();
			//warped_img2.release();
			//cv::copyMakeBorder(warped_img1, temp_frame1, offset1.y, pano_frame_size.height - compose_infos.warped_sizes[0].height - offset1.y, offset1.x, pano_frame_size.width - compose_infos.warped_sizes[0].width - offset1.x, cv::BORDER_CONSTANT, 0.);
			//cv::copyMakeBorder(warped_img2, temp_frame2, offset2.y, pano_frame_size.height - compose_infos.warped_sizes[1].height - offset2.y, offset2.x, pano_frame_size.width - compose_infos.warped_sizes[1].width - offset2.x, cv::BORDER_CONSTANT, 0.);

			//temp_frame1.copyTo(temp_frame2, mask);
			//cv::Mat pano_frame;
			//temp_frame2.copyTo(pano_frame);
			cv::Mat result, pano_frame, result_mask;
			blender->blend(result, result_mask);
			//std::cout << "Marker3" << result.size() << std::endl;
			//img_warped_s1.release();
			//img_warped_s2.release();
			

			cv::Mat pRoi = result(cv::Rect(0, 0, result.size[1], haut));
			pRoi.setTo(cv::Scalar(0));
			pRoi = result(cv::Rect(0, result.size[0] - bas, result.size[1], bas));
			pRoi.setTo(cv::Scalar(0));

			result.convertTo(pano_frame, CV_8U);
			//result.release();
			{
				std::unique_lock<std::mutex> lock(writer_mutex);
				pano_frames.push(pano_frame);
				
				if (frame_idx >= 0 && frame_idx < 250) {
					if (verifId < 10) 
						cv::imwrite("verifPano/" + dirName + "/verif_00" + std::to_string(verifId) + ".jpg", pano_frame);
					else if (verifId < 100)
						cv::imwrite("verifPano/" + dirName + "/verif_0" + std::to_string(verifId) + ".jpg", pano_frame);
					else if (verifId < 1000)
						cv::imwrite("verifPano/" + dirName + "/verif_" + std::to_string(verifId) + ".jpg", pano_frame);
					verifId++;
					verifId = (verifId == 250) ? 0 : verifId;
				}
		
				if (offset_verifPano > 0 && frame_idx >= offset_verifPano && frame_idx < offset_verifPano + 500) {
					if (verifId < 10) 
						cv::imwrite("verifPano/" + dirName + "/verif_00" + std::to_string(verifId) + ".jpg", pano_frame);
					else if (verifId < 100)
						cv::imwrite("verifPano/" + dirName + "/verif_0" + std::to_string(verifId) + ".jpg", pano_frame);
					else if (verifId < 1000)
						cv::imwrite("verifPano/" + dirName + "/verif_" + std::to_string(verifId) + ".jpg", pano_frame);
					verifId++;
				}

				lock.unlock();
				frame_available.notify_one();
			}
        }

        stream1.stop();
        stream2.stop();
        writer_thread.join();

		if(measure_stitching_perfs) {
			auto e2 = cv::getTickCount();
			auto time = (e2 - e1) / cv::getTickFrequency();
			std::cout << "spent " << std::to_string(time / frame_count) << " per frame"
					  << "(" << std::to_string(frame_count / time) << " fps)" << std::endl;
			panoramaVideo.release();
			system("pause");
		}

		return EXIT_SUCCESS;
    }

    StitchingInfos find_stitching_infos(const std::vector<cv::Mat>& full_images, StitchingInfos& prev_stitching_infos)
    {
		size_t num_images = full_images.size();
        cv::Ptr<cvd::FeaturesFinder> finder;
        if (orb_feature_finder)
            finder = cv::makePtr<cvd::OrbFeaturesFinder>();
        else
			finder = cv::makePtr<cvd::AKAZEFeaturesFinder>();
		//finder = cv::makePtr<cvd::SurfFeaturesFinder>();
        std::vector<cvd::ImageFeatures> features(num_images);
		double work_scale = get_scale(full_images, work_megapix);

        for (size_t i = 0; i < num_images; ++i) {
            cv::Mat img;
            const cv::Mat& full_img = full_images[i];
            if (work_megapix < 0)
                img = full_img;
            else
                cv::resize(full_img, img, cv::Size(), work_scale, work_scale, cv::INTER_LINEAR);
			
			// Keeping only center
			cv::Mat pRoi = img(cv::Rect(i == 0 ? 0 : img.size[1] / 2, 0, img.size[1] / 2, img.size[0]));
			// set roi to some rgb colour   
			pRoi.setTo(cv::Scalar(0));
            
			(*finder)(img, features[i]);
            features[i].img_idx = static_cast<int>(i);
        }
        finder->collectGarbage();

        // Pairwise matching
        std::vector<cvd::MatchesInfo> pairwise_matches;
        cv::Ptr<cvd::FeaturesMatcher> matcher = cv::makePtr<cvd::BestOf2NearestRangeMatcher>(range_width, false, match_conf);
        (*matcher)(features, pairwise_matches);
        matcher->collectGarbage();

		// FILTERING MATCHES 1
		std::vector<int> widths; // Distances of matches through x axis
		float median_x = 0;
		if (pairwise_matches[1].matches.size() > 0) {
			widths.resize(pairwise_matches[1].matches.size());
			std::transform(
				pairwise_matches[1].matches.begin(),
				pairwise_matches[1].matches.end(),
				widths.begin(),
				[&](cv::DMatch m) { return std::abs((full_images[0].size().width - features[0].keypoints[m.queryIdx].pt.x) + features[1].keypoints[m.trainIdx].pt.x); });
			std::nth_element(widths.begin(), widths.begin() + widths.size() / 2, widths.end());
			//std::cout << "The median is " << widths.size() || widths[widths.size() / 2] << '\n';
			median_x = widths[widths.size() / 2];
		}
		int idx = 0;
		while (idx < pairwise_matches[1].matches.size())
		{
			cv::DMatch match = pairwise_matches[1].matches[idx];
			float delta_y = features[1].keypoints[match.trainIdx].pt.y - features[0].keypoints[match.queryIdx].pt.y;
			float delta_x = (full_images[0].size().width - features[0].keypoints[match.queryIdx].pt.x) + features[1].keypoints[match.trainIdx].pt.x;
			float delta_border = std::min(std::abs(full_images[0].size().width / 2.0 - features[1].keypoints[match.trainIdx].pt.x),
				std::abs(full_images[0].size().width / 2.0 - features[0].keypoints[match.queryIdx].pt.x));
			//std::cout << "Match delta : " << delta << " : " << idx << " : " << pairwise_matches[1].matches.size() << " , keypoints " << match.imgIdx << " : " << match.queryIdx << " : " << match.trainIdx << std::endl;
			if (std::abs(delta_y) > 120 || std::abs(delta_x) > 1.6 * median_x || delta_border < 10)
			{
				//std::cout << "Match delta : " << delta_y << " : " << features[0].keypoints[match.queryIdx].pt.y << " : " << features[1].keypoints[match.trainIdx].pt.y << " , keypoints " << match.imgIdx << " : " << match.queryIdx << " : " << match.trainIdx << std::endl;
				pairwise_matches[1].matches.erase(pairwise_matches[1].matches.begin() + idx);
				idx--;
			}
			idx++;
		}

		// ADD PREVIOUS FEATURES AND MATCHES
		if (prev_stitching_infos.features.size() != 0) {
			int old_feat1_size = features[1].keypoints.size();
			int old_feat0_size = features[0].keypoints.size();
			//std::cout << "okoko" << prev_stitching_infos.features[1].keypoints.size() << std::endl;
			//std::cout << "okoksizeo" << prev_stitching_infos.pairwise_matches[1].matches.size() << std::endl;
			//std::cout << "okoksizeo" << prev_stitching_infos.features[1].descriptors.size() << std::endl;
			//std::cout << "okoksizeo" << features[1].descriptors.size() << std::endl;
			cv::Mat tempMat1, tempMat0;
			cv::vconcat(features[1].descriptors, prev_stitching_infos.features[1].descriptors, tempMat1);
			tempMat1.copyTo(features[1].descriptors);
			cv::vconcat(features[0].descriptors, prev_stitching_infos.features[0].descriptors, tempMat0);
			tempMat0.copyTo(features[0].descriptors);
			for (int i = 0; i < prev_stitching_infos.features[1].keypoints.size(); i++) {
				features[1].keypoints.push_back(prev_stitching_infos.features[1].keypoints[i]);
			}
			for (int i = 0; i < prev_stitching_infos.features[0].keypoints.size(); i++) {
				features[0].keypoints.push_back(prev_stitching_infos.features[0].keypoints[i]);
			}
			pairwise_matches[0].num_inliers += prev_stitching_infos.pairwise_matches[0].num_inliers;
			pairwise_matches[1].num_inliers += prev_stitching_infos.pairwise_matches[1].num_inliers;
			for (int i = 0; i < prev_stitching_infos.pairwise_matches[1].inliers_mask.size(); i++) {
				pairwise_matches[1].inliers_mask.push_back(prev_stitching_infos.pairwise_matches[1].inliers_mask[i]);
			}
			for (int i = 0; i < prev_stitching_infos.pairwise_matches[0].inliers_mask.size(); i++) {
				pairwise_matches[0].inliers_mask.push_back(prev_stitching_infos.pairwise_matches[0].inliers_mask[i]);
			}
			for (int i = 0; i < prev_stitching_infos.pairwise_matches[1].matches.size(); i++) {
				cv::DMatch currMatch = cv::DMatch(
					prev_stitching_infos.pairwise_matches[1].matches[i].queryIdx + old_feat0_size,
					prev_stitching_infos.pairwise_matches[1].matches[i].trainIdx + old_feat1_size,
					prev_stitching_infos.pairwise_matches[1].matches[i].imgIdx,
					prev_stitching_infos.pairwise_matches[1].matches[i].distance
					);
				pairwise_matches[1].matches.push_back(currMatch);
			}
			for (int i = 0; i < prev_stitching_infos.pairwise_matches[0].matches.size(); i++) {
				cv::DMatch currMatch = cv::DMatch(
					prev_stitching_infos.pairwise_matches[0].matches[i].queryIdx + old_feat1_size,
					prev_stitching_infos.pairwise_matches[0].matches[i].trainIdx + old_feat0_size,
					prev_stitching_infos.pairwise_matches[0].matches[i].imgIdx,
					prev_stitching_infos.pairwise_matches[0].matches[i].distance
				);
				pairwise_matches[1].matches.push_back(currMatch);

				//prev_stitching_infos.pairwise_matches[0].matches[i].queryIdx += old_feat1_size;
				//prev_stitching_infos.pairwise_matches[0].matches[i].trainIdx += old_feat0_size;
				//pairwise_matches[0].matches.push_back(prev_stitching_infos.pairwise_matches[0].matches[i]);
			}
			//std::cout << "end" << prev_stitching_infos.features[1].keypoints.size() << std::endl;

		}
		//std::cout << "vdsjo" << std::endl;
		//std::cout << "vdsjodesc" << features[0].descriptors.size() << std::endl;
		//std::cout << "vdsjoidx" << features[0].img_idx << std::endl;
		//std::cout << "vdsjosize" << features[0].img_size << std::endl;
		//std::cout << "vdsjokp" << features[0].keypoints.size() << std::endl;
		//std::cout << "vdsjo1-------------------" << std::endl;
		//std::cout << "vdsjodesc" << features[1].descriptors.size() << std::endl;
		//std::cout << "vdsjoidx" << features[1].img_idx << std::endl;
		//std::cout << "vdsjosize" << features[1].img_size << std::endl;
		//std::cout << "vdsjokp" << features[1].keypoints.size() << std::endl;
		//std::cout << "pairwise-------------------" << std::endl;
		//std::cout << "vdsjodesc" << pairwise_matches[0].inliers_mask.size() << std::endl;
		//std::cout << "vdsjodesc" << pairwise_matches[0].num_inliers << std::endl;
		//std::cout << "vdsjodesc" << pairwise_matches[0].matches.size() << std::endl;
		//std::cout << "pairwise1-------------------" << std::endl;
		//std::cout << "vdsjodesc" << pairwise_matches[1].inliers_mask.size() << std::endl;
		//std::cout << "vdsjodesc" << pairwise_matches[1].num_inliers << std::endl;
		//std::cout << "vdsjodesc" << pairwise_matches[1].matches.size() << std::endl;

        cv::Ptr<cvd::Estimator> estimator = cv::makePtr<cvd::HomographyBasedEstimator>();
        std::vector<cvd::CameraParams> cameras;
        if (!(*estimator)(features, pairwise_matches, cameras))
            throw std::runtime_error("Homography estimation failed.");

        for (size_t i = 0; i < cameras.size(); ++i) {
            cv::Mat R, K;
            cameras[i].R.convertTo(R, CV_32F);
            cameras[i].R = R;
			//std::cout << "Camera parameters" << cameras[i].K() << std::endl;
			//std::cout << "Camera parameters" << R << std::endl;
		}

        if (draw_matches) {
            static size_t i = 0; // TODO: fix this static variable: make it thread static?
            i++;
            cv::Mat matches;
			// TOERASE
			int maxQuery = 0;
			int maxTrain = 0;
			for (int idx1 = 0; idx1 < pairwise_matches[1].matches.size(); idx1++) {
				if (pairwise_matches[1].matches[idx1].queryIdx > maxQuery) {
					maxQuery = pairwise_matches[1].matches[idx1].queryIdx;
				}
				if (pairwise_matches[1].matches[idx1].trainIdx > maxTrain) {
					maxTrain = pairwise_matches[1].matches[idx1].queryIdx;
				}
			}
            cv::drawMatches(full_images[0], features[0].keypoints, full_images[1], features[1].keypoints, pairwise_matches[1].matches, matches);
				if (i < 10)
						cv::imwrite("matches/" + dirName + "/matches_0" + std::to_string(i) + ".jpg", matches);
				else
						cv::imwrite("matches/" + dirName + "/matches_" + std::to_string(i) + ".jpg", matches);
        }

		cv::Ptr<cvd::BundleAdjusterBase> adjuster = cv::makePtr<cvd::BundleAdjusterReproj>();
		//cv::Ptr<cvd::BundleAdjusterBase> adjuster = cv::makePtr<cvd::BundleAdjusterRay>();
        adjuster->setConfThresh(conf_thresh);
        cv::Mat_<uchar> refine_mask = cv::Mat::zeros(3, 3, CV_8U);
        if (ba_refine_mask[0] == 'x')
            refine_mask(0, 0) = 1;
        if (ba_refine_mask[1] == 'x')
            refine_mask(0, 1) = 1;
        if (ba_refine_mask[2] == 'x')
            refine_mask(0, 2) = 1;
        if (ba_refine_mask[3] == 'x')
            refine_mask(1, 1) = 1;
        if (ba_refine_mask[4] == 'x')
            refine_mask(1, 2) = 1;
        adjuster->setRefinementMask(refine_mask);
		//adjuster->setTermCriteria(cv::TermCriteria(3,1000,0.1));
        if (!(*adjuster)(features, pairwise_matches, cameras))
            throw std::runtime_error("Camera parameters adjusting failed.");

        // Find median focal length
        std::vector<double> focals;
        for (size_t i = 0; i < cameras.size(); ++i) {
            focals.push_back(cameras[i].focal);
        }

        sort(focals.begin(), focals.end());
        float warped_image_scale;
        if (focals.size() % 2 == 1)
            warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
        else
            warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

		//std::cout << "Camera parameters 1" << cameras[0].K() << std::endl;
		//std::cout << "Camera parameters 1" << cameras[0].R << std::endl;

		if (do_wave_correct) {
            std::vector<cv::Mat> cameras_rotation;
            for (size_t i = 0; i < cameras.size(); ++i)
                cameras_rotation.push_back(cameras[i].R);
            cvd::waveCorrect(cameras_rotation, cvd::WAVE_CORRECT_HORIZ);
            for (size_t i = 0; i < cameras.size(); ++i)
                cameras[i].R = cameras_rotation[i];
        }

		//std::cout << "Camera parameters 2" << cameras[0].K() << std::endl;
		//std::cout << "Camera parameters 2" << cameras[0].R << std::endl;
		double score = 0.;
        for (const auto& m : pairwise_matches)
            score += m.confidence * sqrt(m.num_inliers);

		std::vector<cvd::ImageFeatures> saved_features(num_images);
		std::vector<cvd::MatchesInfo> saved_pairwise_matches;

		if (score > 3 * resize_factor *0) {
			saved_features = features;
			saved_pairwise_matches = pairwise_matches;
			score += prev_stitching_infos.score;
		}
		else {
			saved_features = prev_stitching_infos.features;
			saved_pairwise_matches = prev_stitching_infos.pairwise_matches;
		}
		return StitchingInfos(cameras, warped_image_scale, work_scale, score, saved_features, saved_pairwise_matches);
    }

	cv::Ptr<ComposeInfos> find_seam_masks(const std::vector<cv::Mat>& full_images, const StitchingInfos& stitching_infos)
	{
		return find_seam_masks(full_images, stitching_infos, std::vector<cv::Mat>());
	}
	cv::Ptr<ComposeInfos> find_seam_masks(const std::vector<cv::Mat>& full_images, const StitchingInfos& stitching_infos, std::vector<cv::Mat> prev_seamed_mask)
	{
		// Find masks (seam)
		const size_t num_images = full_images.size();
		double seam_scale = get_scale(full_images, seam_megapix);
		std::vector<cv::UMat> seam_masks_warped(num_images);
		std::vector<cv::UMat> images_warped(num_images);
		std::vector<cv::Point> corners(num_images);
		std::vector<cv::UMat> masks(num_images);
		std::vector<cv::Mat> images(num_images);

		for (size_t i = 0; i < num_images; ++i)
			cv::resize(full_images[i], images[i], cv::Size(), seam_scale, seam_scale, cv::INTER_LINEAR);

		// Prepare images masks
		for (size_t i = 0; i < num_images; ++i) {
			masks[i].create(images[i].size(), CV_8U);
			masks[i].setTo(cv::Scalar::all(255));
		}


		// Warp images and their masks
		//std::cout << "ATTENTION WARPER SCALE 3 " << stitching_infos.warped_image_scale * seam_scale / stitching_infos.work_scale << std::endl;

		cv::Ptr<cvd::RotationWarper> warper = create_warper(stitching_infos.warped_image_scale * seam_scale / stitching_infos.work_scale);
		for (size_t i = 0; i < num_images; ++i) {
			cv::Mat_<float> K;
			stitching_infos.cameras[i].K().convertTo(K, CV_32F);
			float swa = (float)(seam_scale / stitching_infos.work_scale);
			K(0, 0) *= swa;
			K(0, 2) *= swa;
			K(1, 1) *= swa;
			K(1, 2) *= swa;

			corners[i] = warper->warp(images[i], K, stitching_infos.cameras[i].R, cv::INTER_LINEAR, cv::BORDER_REFLECT, images_warped[i]);
			//corners[i] = warper->warp(images[i], K, stitching_infos.cameras[i].R, cv::INTER_LINEAR, cv::BORDER_CONSTANT, images_warped[i]);
			warper->warp(masks[i], K, stitching_infos.cameras[i].R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, seam_masks_warped[i]);
		}

		// Prepare images masks
		for (size_t i = 0; i < num_images; ++i) {
			cv::Mat masks_mat;
			masks_mat = seam_masks_warped[i].getMat(cv::ACCESS_READ);
			// Select only region in the middle of the camera
		   // double center_seam_size = 1.0 / CSS1;
			double center_seam_size = 1.0 / css1;
			cv::Mat pRoi = masks_mat(cv::Rect(i == 1 ? 0 : masks_mat.size[1] * (1. - center_seam_size),
				0,
				masks_mat.size[1] * (.0 + center_seam_size),
				masks_mat.size[0]
			));
			pRoi.setTo(cv::Scalar(0));
			seam_masks_warped[i] = masks_mat.getUMat(cv::ACCESS_READ);
		}

		// TODO UPDATE PRE-SEAMFINDER CORNERS SO THAT THE CUT WILL BE IN THE MIDDLE
		std::vector<cv::UMat> images_warped_f(num_images);
		for (size_t i = 0; i < num_images; ++i) {
			cv::medianBlur(images_warped[i], images_warped[i], 9);
			images_warped[i].convertTo(images_warped_f[i], CV_32F);
		}

		//cv::Ptr<cvd::SeamFinder> seam_finder = cv::makePtr<cvd::VoronoiSeamFinder>();
		cv::Ptr<cvd::SeamFinder> seam_finder = cv::makePtr<cvd::GraphCutSeamFinder>(cvd::GraphCutSeamFinderBase::COST_COLOR);
		seam_finder->find(images_warped_f, corners, seam_masks_warped);

		std::vector<cvd::CameraParams> cameras = stitching_infos.cameras;
		std::vector<cv::Size> compose_sizes(num_images);
		std::vector<cv::Point> compose_corners(num_images);
		cv::Mat dilated_mask, seam_mask, mask;
		std::vector<cv::Mat> mask_warped(num_images);
		//auto compose_scale = static_cast<const float>(stitching_infos.warped_image_scale / stitching_infos.work_scale);
		auto compose_scale = 1.0;
		//std::cout << "ATTENTION WARPER SCALE 1 " << stitching_infos.warped_image_scale << std::endl;

		warper = create_warper(stitching_infos.warped_image_scale);

		//std::cout << "Scales : " << "stitching_infos.warped_image_scale * seam_scale / stitching_infos.work_scale" << stitching_infos.warped_image_scale * seam_scale / stitching_infos.work_scale << std::endl;
		//std::cout << "Scales : " << "compose scale" << compose_scale << std::endl;
		//std::cout << "Scales : " << "stitching_infos.work_scale" << stitching_infos.work_scale << std::endl;
		//std::cout << "Scales : " << "seam_scale " << seam_scale << std::endl;
		//std::cout << "Scales : " << "stitching_infos.warped_image_scale" << stitching_infos.warped_image_scale << std::endl;

		std::vector<cv::Mat> cameras_k(num_images), cameras_r(num_images);
		for (size_t img_idx = 0; img_idx < num_images; ++img_idx) {
			const cv::Mat& full_img = full_images[img_idx];

			// Update intrinsics
			cameras[img_idx].focal *= compose_scale / stitching_infos.work_scale;
			cameras[img_idx].ppx *= compose_scale / stitching_infos.work_scale;
			cameras[img_idx].ppy *= compose_scale / stitching_infos.work_scale;


			// Update corner and size
			cv::Size sz = full_images[img_idx].size();
			if (std::abs(compose_scale - 1) > 1e-1) {
				sz.width = cvRound(sz.width * compose_scale);
				sz.height = cvRound(sz.height * compose_scale);
			}
			cv::Mat_<float> K;
			cameras[img_idx].K().convertTo(K, CV_32F);

			//std::cout << "Camera rotation" << sz << std::endl;
			//std::cout << "Camera rotation" << K << std::endl;
			//std::cout << "Camera rotation" << cameras[img_idx].R << std::endl;

			cv::Rect roi = warper->warpRoi(sz, K, cameras[img_idx].R);
			compose_corners[img_idx] = roi.tl();
			compose_sizes[img_idx] = roi.size();
			cameras_k[img_idx] = K;
			cameras[img_idx].R.copyTo(cameras_r[img_idx]);
			//std::cout << "Compose corners" << roi.tl() << std::endl;
			//std::cout << "Compose corners" << roi.br() << std::endl;
			//std::cout << "Compose size" << roi.size() << std::endl;

			// Warp the current image mask
			mask.create(full_img.size(), CV_8UC1);
			mask.setTo(cv::Scalar::all(255));
			warper->warp(mask, cameras_k[img_idx], cameras_r[img_idx], cv::INTER_NEAREST, cv::BORDER_CONSTANT, mask_warped[img_idx]);
			
			// Select only region in the middle of the camera
			double center_seam_size = 1.0/ CSS2;
			cv::Mat pRoi = mask_warped[img_idx](cv::Rect(img_idx == 1 ? 0 : mask_warped[img_idx].size[1] * (1.0-center_seam_size), 
				0, 
				mask_warped[img_idx].size[1] * (0. + center_seam_size), 
				mask_warped[img_idx].size[0]
			));
//			pRoi.setTo(cv::Scalar(0));

			mask.release();
			int erosion_size = 1;
			cv::dilate(seam_masks_warped[img_idx], dilated_mask, cv::getStructuringElement(cv::MORPH_CROSS,
				cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
				cv::Point(erosion_size, erosion_size))
			);
			cv::resize(dilated_mask, seam_mask, mask_warped[img_idx].size(), 0, 0, cv::INTER_LINEAR);
			dilated_mask.release();

			if (prev_seamed_mask.size() == 0) {
				std::cout << "Seam boudary is free like a bird" << std::endl;
			}
			else {
				//std::cout << "Limiting seam variation" << std::endl;
				// Limit seam variation
				cv::Mat eroded_previous, dilated_previous;
				int variation_erosion_size = 4;
				cv::erode(prev_seamed_mask[img_idx], eroded_previous, cv::getStructuringElement(cv::MORPH_CROSS,
					cv::Size(2 * variation_erosion_size + 1, 2 * variation_erosion_size + 1),
					cv::Point(variation_erosion_size, variation_erosion_size)));
				cv::dilate(prev_seamed_mask[img_idx], dilated_previous, cv::getStructuringElement(cv::MORPH_CROSS,
					cv::Size(2 * variation_erosion_size + 1, 2 * variation_erosion_size + 1),
					cv::Point(variation_erosion_size, variation_erosion_size)));
				seam_mask = (seam_mask | eroded_previous ) | (seam_mask & dilated_previous);
			}


			mask_warped[img_idx] = mask_warped[img_idx] & seam_mask;
			seam_mask.release();
		}

		cv::Rect pano_frame_roi = cvd::resultRoi(compose_corners, compose_sizes);
		//std::cout << "Result ROI" << pano_frame_roi.tl() << std::endl;
		cv::Size pano_frame_size = pano_frame_roi.size();

		for (size_t img_idx = 0; img_idx < num_images; ++img_idx) {
			cv::Size compose_size = compose_sizes[img_idx];
			int dx = compose_corners[img_idx].x - pano_frame_roi.x, dy = compose_corners[img_idx].y - pano_frame_roi.y;
			//cv::copyMakeBorder(mask_warped[img_idx], mask_warped[img_idx], dy, pano_frame_size.height - compose_size.height - dy, dx, pano_frame_size.width - compose_size.width - dx, cv::BORDER_CONSTANT, 0.);
		}

		// Make sure masks doesn't overlap
		//assert(num_images == 2); // We assumes there are only two images
		//cv::bitwise_not(mask_warped[0], mask_warped[1]);

		std::vector<cv::Point> offsets = { cv::Point(compose_corners[0].x - pano_frame_roi.x, compose_corners[0].y - pano_frame_roi.y), cv::Point(compose_corners[1].x - pano_frame_roi.x, compose_corners[1].y - pano_frame_roi.y) };
        return cv::makePtr<ComposeInfos>(mask_warped, offsets, pano_frame_roi.size(), compose_sizes, compose_scale, cameras_r, cameras_k, compose_corners);
    }
}

#include <sys/stat.h> 
#include <sys/types.h> 

int main(int argc, const char** argv)
{
	std::cout << "OPENCV version ";
	std::cout << CV_MAJOR_VERSION << ".";
	std::cout << CV_MINOR_VERSION << std::endl;

	mkdir("log", 0777);
	mkdir("matches", 0777);
	stitching::dirName = argv[0];
	mkdir(("matches/" + stitching::dirName).c_str(), 0777);
	mkdir("verifPano", 0777);
	mkdir(("verifPano/" + stitching::dirName).c_str(), 0777);
	mkdir("output", 0777);
	ArgumentParser parser;
	parser.appName("Stitching");
	parser.addArgument("--v1", 1, true);
	parser.addArgument("--v2", 1, true);
	parser.addArgument("-c", "--compose_infos", 1);
	parser.addArgument("-e", "--estimate_infos");
	parser.addArgument("-g", "--left_shift", 1);
	parser.addArgument("-d", "--right_shift", 1);
	parser.addArgument("-t", "--total_shift", 1);
	parser.addArgument("-h", "--haut", 1);
	parser.addArgument("-b", "--bas", 1);
	parser.addArgument("-r", "--rectify_map", 1);
	

	//NEW 
	parser.addArgument("-M", "--match_conf", 1);
	parser.addArgument("-C", "--css1", 1);
	parser.addArgument("-V", "--offset_verifPano", 1);
	//END NEW

	parser.addFinalArgument("output");
	parser.useExceptions(true);

	try
	{
		// Example of arguments usage: "--estimate_infos --v1 "./VID_20180717_093306_0007_cut_1080p.mp4" --v2 "./VID_20180717_093308_0005_1080p.MP4" pano1_1080p.mp4
		parser.parse(argc, argv);

		std::string output = parser.retrieve<std::string>("output"), v1 = parser.retrieve<std::string>("v1"), v2 = parser.retrieve<std::string>("v2");
		std::string compose_infos_path = parser.retrieve<std::string>("compose_infos");
		bool estimate_infos = (compose_infos_path.empty() ? argc - 6 : argc - 8) >= 1; // Parser don't let you know about optionals with 0 args ?!
		
		
		stitching::left_shift = std::stoi(parser.retrieve<std::string>("left_shift"));
		stitching::right_shift = std::stoi(parser.retrieve<std::string>("right_shift"));
		stitching::total_shift = std::stoi(parser.retrieve<std::string>("total_shift"));
		stitching::haut = std::stoi(parser.retrieve<std::string>("haut"));
		stitching::bas = std::stoi(parser.retrieve<std::string>("bas"));
		stitching::rectify_map = std::stoi(parser.retrieve<std::string>("rectify_map"));
		stitching::match_conf = (float)std::stoi(parser.retrieve<std::string>("match_conf")) / 100;
		stitching::css1 = (float)std::stoi(parser.retrieve<std::string>("css1")) / 10;
		stitching::offset_verifPano = std::stoi(parser.retrieve<std::string>("offset_verifPano"));
		
		std::cout << "left shift : " << stitching::left_shift  << std::endl;
		std::cout << "right_shift: " << stitching::right_shift << std::endl;
		std::cout << "total_shift: " << stitching::total_shift << std::endl;
		std::cout << "haut       : " << stitching::haut        << std::endl;
		std::cout << "bas        : " << stitching::bas         << std::endl;
		std::cout << "rectify_map: " << stitching::rectify_map  << std::endl;
		std::cout << "match_conf: " << stitching::match_conf << std::endl;
		std::cout << "css1: " << stitching::css1 << std::endl;
		std::cout << "offset_verifPano: " << stitching::offset_verifPano << std::endl;

		if (!estimate_infos && compose_infos_path.empty()) {
			std::cout << "Can't stitch videos without compose infos." << std::endl;
			return EXIT_FAILURE;
		}
		cv::Ptr<stitching::ComposeInfos> compose_infos;
		if (estimate_infos) {
			compose_infos = stitching::find_compose_infos(v1, v2);
			std::cout << compose_infos->pano_frame_size << "warped sizes" << std::endl;
			if (!compose_infos_path.empty())
				compose_infos->Save(compose_infos_path);
		}
		else if (!compose_infos_path.empty()) {
			std::cout << "Loading compose infos..." << std::endl;
			compose_infos = stitching::ComposeInfos::Load(parser.retrieve<std::string>("compose_infos"));
		}

		if (compose_infos.empty()) {
			std::cout << "Can't estimate compose infos from given videos" << std::endl;
			return EXIT_FAILURE;
		}

		return stitching::stitch_video(v1, v2, *compose_infos, output);
	}
	catch (const std::exception& e) {
		std::cout << "unexcpected exception thrown: " << e.what();
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

