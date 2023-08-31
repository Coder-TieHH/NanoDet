#define _CRT_SECURE_NO_WARNINGS
#include "NanoDet.hpp"
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <sstream>

std::vector<Object> objects;
char fps_show[10] = "FPS:";
int main()
{
	NanoDet_Plus mynet("/home/rpdzkj/Desktop/NanoDet/models/nanodet-plus_320_uint8_tmfile", 320, 0.2, 0.3);
	// string imgpath = "/home/rpdzkj/Desktop/NanoDet/data/1.jpg";
	// Mat srcimg = imread(imgpath);

	cv::VideoCapture capture;
	cv::Mat srcimg;
	capture.open("v4l2src device=/dev/video0 ! video/x-raw, format=RGB, width=1920, height=1080, framerate=1000/30 ! videoconvert ! appsink", cv::CAP_GSTREAMER);

	if (capture.isOpened())
	{
		while (capture.read(srcimg))
		{
			mynet.detect(srcimg, objects);
			for (size_t i = 0; i < objects.size(); i++)
			{
				const Object &obj = objects[i];

				fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
						obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, (class_names[obj.label]).c_str());

				cv::rectangle(srcimg, obj.rect, cv::Scalar(255, 0, 0));

				char text[256];
				sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

				int baseLine = 0;
				cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

				int x = obj.rect.x;
				int y = obj.rect.y - label_size.height - baseLine;
				if (y < 0)
					y = 0;
				if (x + label_size.width > srcimg.cols)
					x = srcimg.cols - label_size.width;

				float fps = 1 / (mynet.top_model_cost + mynet.prepare_cost);
				sprintf(fps_show, "FPS: %.2f", fps);

				cv::rectangle(srcimg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
							  cv::Scalar(255, 255, 255), -1);

				cv::putText(srcimg, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
							cv::Scalar(0, 0, 0));
				cv::putText(srcimg, fps_show, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
			}

			cv::imshow("NanoDet_demo", srcimg);
			cv::waitKey(30);
		}
	}

	return 0;

	// static const string kWinName = "Deep learning object detection in OpenCV";
	// namedWindow(kWinName, WINDOW_NORMAL);
	// imshow(kWinName, srcimg);
	// waitKey(0);
	// destroyAllWindows();
}
