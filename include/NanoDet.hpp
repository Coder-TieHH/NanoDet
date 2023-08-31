#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "types.hpp"
#include "c_api.h"
#include "tengine_operations.h"

using namespace cv;
using namespace std;

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

typedef struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
} Object;

static vector<string> class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

class NanoDet_Plus
{
public:
    NanoDet_Plus(string model_path, int imgsize, float nms_threshold, float objThreshold);
    ~NanoDet_Plus();
    void detect(Mat &cv_image, std::vector<Object> &objects);

    float top_model_cost; // run 的时间
    float prepare_cost;     // 准备的时间
private:
    int init();
    float get_input_data(Mat &img, const float *mean, const float *norm, image &lb, image &pad);
    void get_input_uint8_data(float *input_fp32, uint8_t *input_data, int size, float input_scale, int zero_point);
    void draw_objects(const cv::Mat &bgr, const std::vector<Object> &objects);
    void softmax_(const float *x, float *y, int length);
    void generate_proposal(vector<BoxInfo> &generate_boxes, const float *preds);
    void nms(vector<BoxInfo> &input_boxes);

    float score_threshold = 0.5;
    float nms_threshold = 0.5;

    int num_class;
    int dims[4];


    const bool keep_ratio = false;
    int inpWidth;
    int inpHeight;

    std::vector<uint8_t> input_uint8;
    std::vector<float> input_float;

    float in_scale;
    int in_zp;
    bool init_done;

    std::vector<std::vector<float>> output_float;
    std::vector<float> out_scale;
    std::vector<int> out_zp;

    const int reg_max = 7;
    const int num_stages = 4;
    const int stride[4] = {8, 16, 32, 64};
    const float mean[3] = {103.53, 116.28, 123.675};
    const float norm[3] = {0.017429f, 0.017507f, 0.017125f};
    context_t context;
    graph_t graph;
};
