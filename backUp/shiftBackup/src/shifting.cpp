#include <iostream>
#include <sys/stat.h> 
#include <sys/types.h> 
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/stitching/warpers.hpp"

#include "argparse.h" // Code from https://github.com/hbristow/argparse

namespace shifting {

	int total_shift, left_shift, right_shift;
	

	cv::Mat	print_left_right(cv::Mat left, cv::Mat right) {
	   int size = 500;
	   int x, y;
	   float scale;
	   int max;
	
	   cv::Mat ret = cv::Mat::zeros(cv::Size(75 + 2 * size, 50 + size), CV_8UC3);
	
	   // <=====LEFT =====>
	   if(left.empty()) {
	       std::cout << "Invalid arg in print_left_right : left empty Mat" << std::endl;
	       return cv::Mat();
	   }
	   x = left.cols;
	   y = left.rows;
	   max = (x > y)? x: y;
	   scale = (float) ( (float) max / size );
	   cv::Rect left_roi(25, 25, (int)( x/scale ), (int)( y/scale ));
	   cv::Mat left_tmp;
	   cv::resize(left,left_tmp, cv::Size(left_roi.width, left_roi.height));
	   left_tmp.copyTo(ret(left_roi));
	
	   // <=====RIGHT=====>
	   if(right.empty()) {
	       std::cout << "Invalid arg in print_left_right : right empty Mat" << std::endl;
	       return cv::Mat();
	   }
	   x = right.cols;
	   y = right.rows;
	   max = (x > y)? x: y;
	   scale = (float) ( (float) max / size );
	   cv::Rect right_roi(50 + size, 25, (int)( x/scale ), (int)( y/scale ));
	   cv::Mat right_tmp;
	   cv::resize(right, right_tmp, cv::Size(right_roi.width, right_roi.height));
	   right_tmp.copyTo(ret(right_roi));
		return ret;
	}
	
	char shift(std::string v1, std::string v2) {
		
		std::cout << v1 << std::endl;
		std::cout << v2 << std::endl;
		cv::VideoCapture cap1(v1);
		cv::VideoCapture cap2(v2);
		if (!cap1.isOpened() || !cap2.isOpened()) {
			std::cout << "Can't open given video files." << std::endl;
			return 1;
		}


		//7 sec de shiftingTest sur notre cher MatchDir ;)
		for (size_t i = 0; i < 7 * 30; i++) {
		

		//	cap1.set(cv::CAP_PROP_POS_FRAMES, left_shift + total_shift);
			cap1.set(cv::CAP_PROP_POS_FRAMES, i + left_shift + total_shift);

		
		//	cap2.set(cv::CAP_PROP_POS_FRAMES, 2 * (right_shift + total_shift));
		//	cap2.set(cv::CAP_PROP_POS_FRAMES, 2 * (i + right_shift + total_shift));
		
		//	cap2.set(cv::CAP_PROP_POS_FRAMES, right_shift + total_shift);
			cap2.set(cv::CAP_PROP_POS_FRAMES, i + right_shift + total_shift);
		
			std::cout << "Finding shifting informations from frame#" << i << std::endl;
			cv::Mat left_frame, right_frame;
	
			if (!cap1.read(left_frame) || !cap2.read(right_frame))
				std::cout << "Can't read frame from video capture" << std::endl;
			else {

				cv::flip(left_frame, left_frame, -1);
				//cv::flip(right_frame, right_frame, -1);

				if (i < 10)
						cv::imwrite("shifting/frame_00" + std::to_string(i) + ".jpg", print_left_right(left_frame, right_frame));
				else if (i < 100)
						cv::imwrite("shifting/frame_0" + std::to_string(i) + ".jpg", print_left_right(left_frame, right_frame));
				else
						cv::imwrite("shifting/frame_" + std::to_string(i) + ".jpg", print_left_right(left_frame, right_frame));
			}
		}
		return 1;
	}
}

int		main(int argc, const char **argv) {
	std::cout << "OPENCV version ";
	std::cout << CV_MAJOR_VERSION << ".";
	std::cout << CV_MINOR_VERSION << std::endl;
	
	ArgumentParser parser;
	parser.appName("Shifting");
	parser.addArgument("--v1", 1, true);
	parser.addArgument("--v2", 1, true);
	parser.addArgument("-g", "--left_shift", 1);
	parser.addArgument("-d", "--right_shift", 1);
	parser.addArgument("-t", "--total_shift", 1);
	mkdir("shifting", 0777);

	

	try {
		parser.parse(argc, argv);
		
		std::string v1, v2;
		v1 = parser.retrieve<std::string>("v1"); 
		v2 = parser.retrieve<std::string>("v2");
		
		shifting::left_shift = std::stoi(parser.retrieve<std::string>("left_shift"));
		shifting::right_shift = std::stoi(parser.retrieve<std::string>("right_shift"));
		shifting::total_shift = std::stoi(parser.retrieve<std::string>("total_shift"));
	
		short frame, min, sec;
		frame = shifting::left_shift;
		sec = frame / 30;
		min = sec / 60;
		sec = sec % 60;
		std::cout << "left_shift : " << shifting::left_shift << " frames" << " or " << min << "min" << sec << "sec" << std::endl;
		frame = shifting::right_shift;
		sec = frame / 30;
		min = sec / 60;
		sec = sec % 60;
		std::cout << "right_shift : " << shifting::right_shift << " frames" << " or " << min << "min" << sec << "sec" << std::endl;
		frame = shifting::total_shift;
		sec = frame / 30;
		min = sec / 60;
		sec = sec % 60;
		std::cout << "total_shift : " << shifting::total_shift << " frames" << " or " << min << "min" << sec << "sec" << std::endl;
	
	shifting::shift(v1, v2);
	}
	catch (const std::exception& e) {
		std::cout << "unexcpected exception thrown: " << e.what();
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;

}
