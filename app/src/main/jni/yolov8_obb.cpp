// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// pip3 install -U ultralytics pnnx ncnn
// yolo export model=yolov8n-obb.pt format=torchscript
// pnnx yolov8n-obb.torchscript
// modify yolov8n_obb_pnnx.py
// python3 -c 'import yolov8n_obb_pnnx; yolov8n_obb_pnnx.export_torchscript()'
// pnnx yolov8n_obb_pnnx.py.pt inputshape=[1,3,1024,1024] inputshape2=[1,3,512,512]

// yolov8n_obb_pnnx.py.ncnn.param
// yolov8n_obb_pnnx.py.ncnn.bin

// the out blob would be a 2-dim tensor with w=79 h=21504
//
//        | bbox-reg 16 x 4       |score(15)|
//        +-----+-----+-----+-----+---------+
//        | dx0 | dy0 | dx1 | dy1 | 0.1 ... |
//   all /|     |     |     |     |     ... |
//  boxes |  .. |  .. |  .. |  .. | 0.0 ... |
// (21504)|     |     |     |     |  .  ... |
//       \|     |     |     |     |  .  ... |
//        +-----+-----+-----+-----+---------+
//

// the out blob would be a 2-dim tensor with w=1 h=21504
//
//        | degree(1)|
//        +----------+
//        |    0.1   |
//   all /|          |
//  boxes |    0.0   |
// (21504)|     .    |
//       \|     .    |
//        +----------+
//

#include "yolov8.h"

#include "layer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <float.h>
#include <stdio.h>
#include <vector>

static inline float intersection_area(const Object& a, const Object& b)
{
    std::vector<cv::Point2f> intersection;
    cv::rotatedRectangleIntersection(a.rrect, b.rrect, intersection);
    if (intersection.empty())
        return 0.f;

    return cv::contourArea(intersection);
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    // #pragma omp parallel sections
    {
        // #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        // #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rrect.size.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void generate_proposals(const ncnn::Mat& pred, const ncnn::Mat& pred_angle, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;

    const int reg_max_1 = 16;
    const int num_class = pred.w - reg_max_1 * 4; // number of classes. 15 for DOTAv1

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            {
                const ncnn::Mat pred_score = pred_grid.range(reg_max_1 * 4, num_class);

                for (int k = 0; k < num_class; k++)
                {
                    float s = pred_score[k];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }

                score = sigmoid(score);
            }

            if (score >= prob_threshold)
            {
                ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4).clone();

                {
                    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1); // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);

                    softmax->forward_inplace(pred_bbox, opt);

                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = pred_bbox.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }

                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (x + 0.5f) * stride;
                float pb_cy = (y + 0.5f) * stride;

                const float angle = sigmoid(pred_angle.row(y * num_grid_x + x)[0]) - 0.25f;

                const float angle_rad = angle * 3.14159265358979323846f;
                const float angle_degree = angle * 180.f;

                float cos = cosf(angle_rad);
                float sin = sinf(angle_rad);

                float xx = (pred_ltrb[2] - pred_ltrb[0]) * 0.5f;
                float yy = (pred_ltrb[3] - pred_ltrb[1]) * 0.5f;
                float xr = xx * cos - yy * sin;
                float yr = xx * sin + yy * cos;
                const float cx = pb_cx + xr;
                const float cy = pb_cy + yr;
                const float ww = pred_ltrb[2] + pred_ltrb[0];
                const float hh = pred_ltrb[3] + pred_ltrb[1];

                Object obj;
                obj.rrect = cv::RotatedRect(cv::Point2f(cx, cy), cv::Size_<float>(ww, hh), angle_degree);
                obj.label = label;
                obj.prob = score;

                objects.push_back(obj);
            }
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const ncnn::Mat& pred_angle, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;

    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++)
    {
        const int stride = strides[i];

        const int num_grid_x = w / stride;
        const int num_grid_y = h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        generate_proposals(pred.row_range(pred_row_offset, num_grid), pred_angle.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects);

        pred_row_offset += num_grid;
    }
}

int YOLOv8_obb::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    const int target_size = det_target_size;//1024;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // ultralytics/cfg/models/v8/yolov8.yaml
    std::vector<int> strides(3);
    strides[0] = 8;
    strides[1] = 16;
    strides[2] = 32;
    const int max_stride = 32;

    // letterbox pad to multiple of max_stride
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    // letterbox pad to target_size rectangle
    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov8.create_extractor();

    ex.input("in0", in_pad);

    ncnn::Mat out;
    ex.extract("out0", out);

    ncnn::Mat out_angle;
    ex.extract("out1", out_angle);

    std::vector<Object> proposals;
    generate_proposals(out, out_angle, strides, in_pad, prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    if (count == 0)
        return 0;

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        Object obj = proposals[picked[i]];

        // adjust offset to original unpadded
        obj.rrect.center.x = (obj.rrect.center.x - (wpad / 2)) / scale;
        obj.rrect.center.y = (obj.rrect.center.y - (hpad / 2)) / scale;
        obj.rrect.size.width = (obj.rrect.size.width) / scale;
        obj.rrect.size.height = (obj.rrect.size.height) / scale;

        objects[i] = obj;
    }

    return 0;
}

int YOLOv8_obb::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "plane", "ship", "storage tank", "baseball diamond", "tennis court",
        "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
        "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"
    };

    static const cv::Scalar colors[] = {
        cv::Scalar( 39, 176, 156),
        cv::Scalar( 58, 183, 103),
        cv::Scalar( 81, 181,  63),
        cv::Scalar(150, 243,  33),
        cv::Scalar(169, 244,   3),
        cv::Scalar(188, 212,   0),
        cv::Scalar(150, 136,   0),
        cv::Scalar(175,  80,  76),
        cv::Scalar(195,  74, 139),
        cv::Scalar(220,  57, 205),
        cv::Scalar(235,  59, 255),
        cv::Scalar(193,   7, 255),
        cv::Scalar(152,   0, 255),
        cv::Scalar( 87,  34, 255),
        cv::Scalar( 85,  72, 121),
        cv::Scalar(158, 158, 158),
        cv::Scalar(125, 139,  96)
    };

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[obj.label];

        // fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f  @ %.2f\n", obj.label, obj.prob,
                // obj.rrect.center.x, obj.rrect.center.y, obj.rrect.size.width, obj.rrect.size.height, obj.rrect.angle);

        cv::Point2f corners[4];
        obj.rrect.points(corners);
        cv::line(rgb, corners[0], corners[1], color);
        cv::line(rgb, corners[1], corners[2], color);
        cv::line(rgb, corners[2], corners[3], color);
        cv::line(rgb, corners[3], corners[0], color);
    }

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const cv::Scalar& color = colors[obj.label];

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rrect.center.x - label_size.width / 2;
        int y = obj.rrect.center.y - label_size.height / 2 - baseLine;
        if (y < 0)
            y = 0;
        if (y + label_size.height > rgb.rows)
            y = rgb.rows - label_size.height;
        if (x < 0)
            x = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}