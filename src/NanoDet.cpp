#include "NanoDet.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include "timer.hpp"
#include "common.h"

using namespace cv;
using namespace std;

NanoDet_Plus::NanoDet_Plus(const string model_path, const string classesFile, int imgsize, float nms_threshold, float objThreshold)
{
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line))
        this->class_names.push_back(line);
    this->num_class = class_names.size();
    this->nms_threshold = nms_threshold;
    this->score_threshold = objThreshold;
    this->init_done = false;

    this->inpHeight = imgsize;
    this->inpWidth = imgsize;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create VeriSilicon TIM-VX backend */
    this->context = create_context("timvx", 1);
    set_context_device(this->context, "TIMVX", nullptr, 0);

    this->graph = create_graph(this->context, "tengine", model_path.c_str());
    if (nullptr == this->graph)
    {
        fprintf(stderr, "Load Tengine model failed.\n");
    }

    tensor_t input_tensor = get_graph_input_tensor(this->graph, 0, 0);
    if (nullptr == input_tensor)
    {
        fprintf(stderr, "Get input tensor failed.\n");
    }

    get_tensor_quant_param(input_tensor, &this->in_scale, &this->in_zp, 1);
}

// @brief:  get input data, keep aspect ratio and fill to the center of letter box
// @param:  lb[in/out]  letter box image inst
// @param:  pad[out]    top and left pad size
// @return: resize scale from origin image to letter box
float NanoDet_Plus::get_input_data(Mat &img, const float *mean, const float *norm, image &lb, image &pad)
{
    if (img.empty())
    {
        fprintf(stderr, "cv::imread failed\n");
        return -1.;
    }
    /* letterbox process to support different letterbox size */
    float lb_scale_w = lb.w * 1. / img.cols, lb_scale_h = lb.h * 1. / img.rows;
    float lb_scale = lb_scale_w < lb_scale_h ? lb_scale_w : lb_scale_h;
    int w = lb_scale * img.cols;
    int h = lb_scale * img.rows;

    if (w != lb.w || h != lb.h)
    {
        cv::resize(img, img, cv::Size(w, h));
    }

    img.convertTo(img, CV_32FC3);

    // Pad to letter box rectangle
    pad.w = lb.w - w; //(w + 31) / 32 * 32 - w;
    pad.h = lb.h - h; //(h + 31) / 32 * 32 - h;
    // Generate a gray image using opencv
    cv::Mat img_pad(lb.w, lb.h, CV_32FC3, // cv::Scalar(0));
                    cv::Scalar(0.5 / norm[0] + mean[0], 0.5 / norm[0] + mean[0], 0.5 / norm[2] + mean[2]));
    // Letterbox filling
    cv::copyMakeBorder(img, img_pad, pad.h / 2, pad.h - pad.h / 2, pad.w / 2, pad.w - pad.w / 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    img_pad.convertTo(img_pad, CV_32FC3);
    float *_data = (float *)img_pad.data;
    /* nhwc to nchw */
    for (int h = 0; h < lb.h; h++)
    {
        for (int w = 0; w < lb.w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * lb.w * 3 + w * 3 + c;
                int out_index = c * lb.h * lb.w + h * lb.w + w;
                lb.data[out_index] = (_data[in_index] - mean[c]) * norm[c];
            }
        }
    }

    return lb_scale;
}

void NanoDet_Plus::get_input_uint8_data(float *input_fp32, uint8_t *input_data, int size, float input_scale, int zero_point)
{
    for (int i = 0; i < size; i++)
    {
        int udata = (round)(input_fp32[i] / input_scale + zero_point);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;

        input_data[i] = udata;
    }
}

void NanoDet_Plus::softmax_(const float *x, float *y, int length)
{
    float sum = 0;
    int i = 0;
    for (i = 0; i < length; i++)
    {
        y[i] = exp(x[i]);
        sum += y[i];
    }
    for (i = 0; i < length; i++)
    {
        y[i] /= sum;
    }
}

void NanoDet_Plus::generate_proposal(vector<BoxInfo> &generate_boxes, const float *preds)
{
    const int reg_1max = reg_max + 1;
    const int len = this->num_class + 4 * reg_1max;
    for (int n = 0; n < this->num_stages; n++)
    {
        const int stride_ = this->stride[n];
        const int num_grid_y = (int)ceil((float)this->inpHeight / stride_);
        const int num_grid_x = (int)ceil((float)this->inpWidth / stride_);
        ////cout << "num_grid_x=" << num_grid_x << ",num_grid_y=" << num_grid_y << endl;

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                int max_ind = 0;
                float max_score = 0;
                for (int k = 0; k < num_class; k++)
                {
                    if (preds[k] > max_score)
                    {
                        max_score = preds[k];
                        max_ind = k;
                    }
                }
                if (max_score >= score_threshold)
                {
                    const float *pbox = preds + this->num_class;
                    float dis_pred[4];
                    float *y = new float[reg_1max];
                    for (int k = 0; k < 4; k++)
                    {
                        softmax_(pbox + k * reg_1max, y, reg_1max);
                        float dis = 0.f;
                        for (int l = 0; l < reg_1max; l++)
                        {
                            dis += l * y[l];
                        }
                        dis_pred[k] = dis * stride_;
                    }
                    delete[] y;
                    /*float pb_cx = (j + 0.5f) * stride_ - 0.5;
                    float pb_cy = (i + 0.5f) * stride_ - 0.5;*/
                    float pb_cx = j * stride_;
                    float pb_cy = i * stride_;
                    float x0 = pb_cx - dis_pred[0];
                    float y0 = pb_cy - dis_pred[1];
                    float x1 = pb_cx + dis_pred[2];
                    float y1 = pb_cy + dis_pred[3];
                    generate_boxes.push_back(BoxInfo{x0, y0, x1, y1, max_score, max_ind});
                }
                preds += len;
            }
        }
    }
}

void NanoDet_Plus::nms(vector<BoxInfo> &input_boxes)
{
    sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b)
         { return a.score > b.score; });
    vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }

    vector<bool> isSuppressed(input_boxes.size(), false);
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < int(input_boxes.size()); ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }
            float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

            float w = (max)(float(0), xx2 - xx1 + 1);
            float h = (max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);

            if (ovr >= this->nms_threshold)
            {
                isSuppressed[j] = true;
            }
        }
    }
    // return post_nms;
    int idx_t = 0;
    input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo &f)
                                { return isSuppressed[idx_t++]; }),
                      input_boxes.end());
}

void NanoDet_Plus::detect(Mat &srcimg)
{

    if (nullptr == this->graph)
    {
        fprintf(stderr, "Graph was not ready.\n");
        return;
    }

    /* initial the graph and prerun graph */
    if (!this->init_done)
    {
        int ret = this->init();
        if (0 != ret)
        {
            fprintf(stderr, "Init graph failed.\n");
            return;
        }
        this->init_done = true;
#ifdef DEBUG
        dump_graph(this->graph);
#endif
    }
    /* prepare process, letterbox */
    Timer prepare_timer;

    image lb = make_image(dims[3], dims[2], dims[1]);
    int img_size = lb.w * lb.h * lb.c;
    image pad = make_empty_image(lb.w, lb.h, lb.c);
    float lb_scale = get_input_data(srcimg, mean, norm, lb, pad);
    get_input_uint8_data(lb.data, this->input_uint8.data(), img_size, this->in_scale, this->in_zp);
    float prepare_cost = (float)prepare_timer.Cost();
    fprintf(stdout, "Prepare cost   %.2fms.\n", prepare_cost);
    // 预处理结束

    /* network inference */
    Timer model_timer;
    int ret = run_graph(this->graph, 1);
    float top_model_cost = (float)model_timer.Cost();
    fprintf(stdout, "Run graph cost %.2fms.\n", top_model_cost);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph failed.\n");
        return;
    }

    /////generate proposals
    vector<BoxInfo> generate_boxes;
    tensor_t p_output = get_graph_output_tensor(graph, 0, 0);
    int output_count = get_tensor_buffer_size(p_output) / sizeof(uint8_t);
    float output_scale = 0.f;
    int output_zp = 0;
    get_tensor_quant_param(p_output, &output_scale, &output_zp, 1);
    uint8_t *p_data = (uint8_t *)get_tensor_buffer(p_output);
    // std::vector<float> out_data(output_count);
    float *out_data = new float[output_count];
    memset(out_data, 0, output_count * sizeof(float));
    for (int c = 0; c < output_count; c++)
    {
        out_data[c] = ((float)p_data[c] - (float)output_zp) * output_scale;
    }

    generate_proposal(generate_boxes, out_data);
    delete[] out_data;

    //// Perform non maximum suppression to eliminate redundant overlapping boxes with
    //// lower confidences
    nms(generate_boxes);
    for (size_t i = 0; i < generate_boxes.size(); ++i)
    {
        int xmin = (int)max((generate_boxes[i].x1 - (pad.w / 2)) / lb_scale, 0.f);
        int ymin = (int)max((generate_boxes[i].y1 - (pad.h / 2)) / lb_scale, 0.f);
        int xmax = (int)min((generate_boxes[i].x2 - (pad.w / 2)) / lb_scale, (float)srcimg.cols);
        int ymax = (int)min((generate_boxes[i].y2 - (pad.h / 2)) / lb_scale, (float)srcimg.rows);
        rectangle(srcimg, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);
        string label = format("%.2f", generate_boxes[i].score);
        label = this->class_names[generate_boxes[i].label] + ":" + label;
        putText(srcimg, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
    }
}

int NanoDet_Plus::init()
{
    /* set runtime options */
    opt.num_thread = 1;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    Timer timer;
    this->input_uint8.resize(this->inpWidth * this->inpHeight * 3);

    tensor_t input_tensor = get_graph_input_tensor(this->graph, 0, 0);
    if (nullptr == input_tensor)
    {
        fprintf(stderr, "Get input tensor was failed.\n");
        return -1;
    }

    this->dims[0] = 1;
    this->dims[1] = 3;
    this->dims[2] = this->inpHeight;
    this->dims[3] = this->inpWidth;

    int ret = set_tensor_shape(input_tensor, dims, 4);
    if (0 != ret)
    {
        fprintf(stderr, "Set input tensor shape failed.\n");
        return -1;
    }

    ret = set_tensor_buffer(input_tensor, this->input_uint8.data(), this->input_uint8.size());
    if (0 != ret)
    {
        fprintf(stderr, "Set input tensor buffer failed.\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    auto time_cost = (float)timer.Cost();
    fprintf(stdout, "Init cost %.2fms.\n", time_cost);

    return 0;
}

NanoDet_Plus::~NanoDet_Plus()
{
    postrun_graph(this->graph);
    destroy_graph(this->graph);
    destroy_context(this->context);
}