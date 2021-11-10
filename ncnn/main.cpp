#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "gpu.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <algorithm>

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;
ncnn::Net detector_;

int main()
{
    detector_.opt.use_packing_layout = true;
    detector_.opt.use_bf16_storage = true;
    detector_.opt.lightmode = true;
    detector_.opt.blob_allocator = &g_blob_pool_allocator;
    detector_.opt.workspace_allocator = &g_workspace_pool_allocator;
    detector_.load_param("../yoloface-opt.param");
    detector_.load_model("../yoloface-opt.bin");

    cv::Mat img = cv::imread("../yoloface-50k-1.jpg");

    int img_w = img.cols;
    int img_h = img.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB,\
                                                img_w, img_h, 56, 56);

    //数据预处理
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = detector_.create_extractor();
    ex.set_num_threads(3);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);
    for (int i = 0; i < out.h; i++)
    {
        int label;
        float x1, y1, x2, y2, score;
        const float* values = out.row(i);
        
        x1 = values[2] * img_w;
        y1 = values[3] * img_h;
        x2 = values[4] * img_w;
        y2 = values[5] * img_h;

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0,0,255));
    }
    cv::imshow("img", img);
    cv::waitKey(0);
    return 0;
}
