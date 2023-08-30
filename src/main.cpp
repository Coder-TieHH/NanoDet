#define _CRT_SECURE_NO_WARNINGS
#include "NanoDet.hpp"
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int main()
{
	NanoDet_Plus mynet("/home/rpdzkj/Desktop/NanoDet/models/nanodet-plus_320_uint8_tmfile", "src/coco.names", 320, 0.5, 0.5);
	/// choice = ["onnxmodel/nanodet-plus-m_320.onnx", "onnxmodel/nanodet-plus-m_416.onnx", "onnxmodel/nanodet-plus-m-1.5x_320.onnx", "onnxmodel/nanodet-plus-m-1.5x_416.onnx"]
	string imgpath = "/home/rpdzkj/Desktop/NanoDet/data/0.jpg";
	Mat srcimg = imread(imgpath);
	mynet.detect(srcimg);

	cv::imwrite("./detect/1", srcimg);
	return 0;

	// static const string kWinName = "Deep learning object detection in OpenCV";
	// namedWindow(kWinName, WINDOW_NORMAL);
	// imshow(kWinName, srcimg);
	// waitKey(0);
	// destroyAllWindows();
}
