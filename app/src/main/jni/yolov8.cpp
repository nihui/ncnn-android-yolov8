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

#include "yolov8.h"

YOLOv8::~YOLOv8()
{
    det_target_size = 320;
}

int YOLOv8::load(const char* parampath, const char* modelpath, bool use_gpu)
{
    yolov8.clear();

    yolov8.opt = ncnn::Option();

#if NCNN_VULKAN
    yolov8.opt.use_vulkan_compute = use_gpu;
#endif

    yolov8.load_param(parampath);
    yolov8.load_model(modelpath);

    return 0;
}

int YOLOv8::load(AAssetManager* mgr, const char* parampath, const char* modelpath, bool use_gpu)
{
    yolov8.clear();

    yolov8.opt = ncnn::Option();

#if NCNN_VULKAN
    yolov8.opt.use_vulkan_compute = use_gpu;
#endif

    yolov8.load_param(mgr, parampath);
    yolov8.load_model(mgr, modelpath);

    return 0;
}

void YOLOv8::set_det_target_size(int target_size)
{
    det_target_size = target_size;
}
