#include "opencv2/core.hpp"

#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <ctime>

#define THRESHOLD 			1.0
#define	SLICE_IN_SEC		30


void resetStringStream(std::stringstream& stream) {
    const static std::stringstream initial;

    stream.str(std::string());
    stream.clear();
    stream.copyfmt(initial);
}

class	Disturb {
private:			
			std::ofstream 					_log;
			double							_fps;
			std::stringstream 			_ss;
			int								_s, _m, _h, _f_start;
			double							_radius;



public:
		
			Disturb( void ) {
			}
			Disturb( std::string logFileName, double fps ) : _log(logFileName), _fps(fps) {
				_h = _m = _s = _f_start = 0;
				_radius = 0.0;
			}
			~Disturb( void ) {
			}

			void						printSettings( void ) {
				_log << "FPS = " << _fps << std::endl;
				_log << "THRESHOLD = " << THRESHOLD << std::endl;
				_log << "SLICE IN SECONDE = " << SLICE_IN_SEC << std::endl;
			}

			std::string				time(int f) {
				std::string		t;

				_h = f / (3600 * _fps);
				_m = f / (60 * _fps) - 60 * _h;
				_s = f / _fps - (3600 * _h + 60 * _m);
				_ss << _h << ":" << _m << ":" << _s;
				t = _ss.str();
				resetStringStream(_ss);
				return (t);
			}

			void						update( double radius, int f ) {

					if (f % (int)(5 * 60 * _fps) == 0)
						_log << "\033[36m" << time(f) << "\033[0m" << std::endl;
				
					if (_f_start == 0 && radius >= THRESHOLD) {
						_f_start = f;
						_radius = radius;
						_log << "\033[37m" << time(f) <<  "\033[0m\033[31m -> DISTURBANCE BEGINS\033[0m" << std::endl;
					}
					else if (_f_start != 0 && radius >= THRESHOLD) {
						_f_start = f;
						_radius = (radius > _radius) ? radius : _radius;
					}
					else if (_f_start != 0 && radius < THRESHOLD && f - _f_start  > SLICE_IN_SEC * _fps) {
						_log <<  "\033[33m        pic recorded: " << _radius << "\033[0m" << std::endl;
						_log << "\033[37m" << time(f) <<  "\033[0m\033[32m -> DISTURBANCE ENDS\033[0m" << std::endl;
						_f_start = radius = 0;
					}

			}
};


using namespace cv;

int main(int argc, char* argv[])
{
		if (argc != 2) {
			std::cout << "usage: ./detectVideoDisturb file.mp4" << std::endl;
			return 0;
		}

   	std::string				input(argv[1]);
		VideoCapture 			video(input);
		Disturb					disturb(input.replace(input.size() - 3, 3, "vDist.log"), video.get(CAP_PROP_FPS));

		disturb.printSettings();

		Mat 						frame, curr, prev, curr64f, prev64f, hann;
		char						key = 0;
		int						f = 0;

    do
    {		
			video >> frame;
			if (frame.empty()) {
				//std::cout << "EMPTY" << std::endl;
				return 0;
			}
        
			resize(frame, frame, Size(640/3, 480/3));
			cvtColor(frame, curr, COLOR_RGB2GRAY);	
			
			if(prev.empty()) {
				prev = curr.clone();
				createHanningWindow(hann, curr.size(), CV_64F);
			}

			prev.convertTo(prev64f, CV_64F);
			curr.convertTo(curr64f, CV_64F);

			Point2d shift = phaseCorrelate(prev64f, curr64f, hann);
			double radius = std::sqrt(shift.x*shift.x + shift.y*shift.y);

	//		if(radius > THRESHOLD) {
	//			// draw a circle and line indicating the shift direction...
	//			Point center(curr.cols >> 1, curr.rows >> 1);
	//			circle(frame, center, (int)radius, Scalar(0, 255, 0), 3, LINE_AA);
	//			line(frame, center, Point(center.x + (int)shift.x, center.y + (int)shift.y), Scalar(0, 255, 0), 3, LINE_AA);
	//		}
			disturb.update(radius, f);

		//	imshow(input, frame);
		//	key = (char)waitKey(2);	
			prev = curr.clone();
			f++;
				
    } while(key != 27);

    return 0;
}
