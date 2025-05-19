// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/bitmap.h> // For Bitmap operations
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolov8.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat& rgb)
{
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static YOLOv8* g_yolov8 = 0;
static ncnn::Mutex lock;

// For getCurrentFrame
static cv::Mat g_latest_frame_for_bitmap;
static ncnn::Mutex g_latest_frame_lock;


class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // yolov8
    {
        ncnn::MutexLockGuard g(lock); // This lock is for g_yolov8

        if (g_yolov8)
        {
            std::vector<Object> objects;
            g_yolov8->detect(rgb, objects);
            g_yolov8->draw(rgb, objects);
        }
        else
        {
            draw_unsupported(rgb);
        }
    }

    draw_fps(rgb);

    // Store the latest frame for bitmap retrieval
    {
        ncnn::MutexLockGuard lock(g_latest_frame_lock);
        g_latest_frame_for_bitmap = rgb.clone(); // Make a deep copy
    }
}

static MyNdkCamera* g_camera = 0;

// Helper function to convert cv::Mat (RGB or RGBA) to Android Bitmap
jobject matToBitmap(JNIEnv* env, const cv::Mat& src) {
    if (src.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "matToBitmap: Input Mat is empty");
        return nullptr;
    }

    // Get Bitmap class and createBitmap method ID
    jclass bitmapCls = env->FindClass("android/graphics/Bitmap");
    if (!bitmapCls) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to find Bitmap class");
        return nullptr;
    }
    jmethodID createBitmapMethod = env->GetStaticMethodID(bitmapCls, "createBitmap", "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    if (!createBitmapMethod) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to find createBitmap static method");
        env->DeleteLocalRef(bitmapCls);
        return nullptr;
    }

    // Get Bitmap.Config.ARGB_8888
    jclass bitmapConfigCls = env->FindClass("android/graphics/Bitmap$Config");
    if (!bitmapConfigCls) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to find Bitmap.Config class");
        env->DeleteLocalRef(bitmapCls);
        return nullptr;
    }
    jfieldID argb8888Field = env->GetStaticFieldID(bitmapConfigCls, "ARGB_8888", "Landroid/graphics/Bitmap$Config;");
    if (!argb8888Field) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to find ARGB_8888 field");
        env->DeleteLocalRef(bitmapCls);
        env->DeleteLocalRef(bitmapConfigCls);
        return nullptr;
    }
    jobject argb8888Config = env->GetStaticObjectField(bitmapConfigCls, argb8888Field);

    // Create the Java Bitmap object
    jobject javaBitmap = env->CallStaticObjectMethod(bitmapCls, createBitmapMethod, src.cols, src.rows, argb8888Config);
    if (!javaBitmap) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to create Java Bitmap object");
        env->DeleteLocalRef(bitmapCls);
        env->DeleteLocalRef(bitmapConfigCls);
        env->DeleteLocalRef(argb8888Config);
        return nullptr;
    }

    // Lock pixels for writing
    void* bitmapPixels;
    if (AndroidBitmap_lockPixels(env, javaBitmap, &bitmapPixels) < 0) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Failed to lock Bitmap pixels");
        // Note: In case of failure, javaBitmap might need to be recycled on Java side if it was created.
        // However, if lockPixels fails, it's safer to return null and let Java handle it.
        env->DeleteLocalRef(bitmapCls);
        env->DeleteLocalRef(bitmapConfigCls);
        env->DeleteLocalRef(argb8888Config);
        // javaBitmap is not deleted here as it might be a valid object that simply couldn't be locked.
        // However, without pixels, it's not useful. Better to ensure it's cleaned up if we return null.
        // For simplicity now, we'll assume if lockPixels fails, the bitmap isn't fully usable.
        return nullptr;
    }

    cv::Mat mat_rgba;
    if (src.channels() == 3) { // RGB
        cv::cvtColor(src, mat_rgba, cv::COLOR_RGB2RGBA);
    } else if (src.channels() == 4) { // Assuming RGBA already
        mat_rgba = src;
    } else {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "Unsupported Mat channels for bitmap: %d", src.channels());
        AndroidBitmap_unlockPixels(env, javaBitmap);
        env->DeleteLocalRef(bitmapCls);
        env->DeleteLocalRef(bitmapConfigCls);
        env->DeleteLocalRef(argb8888Config);
        // Consider deleting/recycling javaBitmap if returning null
        return nullptr;
    }

    // Copy data. Android ARGB_8888 expects RGBA byte order in memory.
    memcpy(bitmapPixels, mat_rgba.data, mat_rgba.total() * mat_rgba.elemSize());

    // Unlock pixels
    AndroidBitmap_unlockPixels(env, javaBitmap);

    // Clean up local JNI references
    env->DeleteLocalRef(bitmapCls);
    env->DeleteLocalRef(bitmapConfigCls);
    env->DeleteLocalRef(argb8888Config);

    return javaBitmap;
}


extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete g_yolov8;
        g_yolov8 = 0;
    }
    {
        ncnn::MutexLockGuard g(g_latest_frame_lock);
        g_latest_frame_for_bitmap.release();
    }


    ncnn::destroy_gpu_instance();

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int taskid, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_loadModel(JNIEnv* env, jobject thiz, jobject assetManager, jint taskid, jint modelid, jint cpugpu)
{
    if (taskid < 0 || taskid > 5 || modelid < 0 || modelid > 8 || cpugpu < 0 || cpugpu > 2)
    {
        return JNI_FALSE;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* tasknames[6] =
            {
                    "",
                    "_oiv7",
                    "_seg",
                    "_pose",
                    "_cls",
                    "_obb"
            };

    const char* modeltypes[9] =
            {
                    "n",
                    "s",
                    "m",
                    "n",
                    "s",
                    "m",
                    "n",
                    "s",
                    "m"
            };

    std::string parampath = std::string("yolov8") + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.param";
    std::string modelpath = std::string("yolov8") + modeltypes[(int)modelid] + tasknames[(int)taskid] + ".ncnn.bin";
    bool use_gpu = (int)cpugpu == 1;
    bool use_turnip = (int)cpugpu == 2;

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        {
            static int old_taskid = 0;
            static int old_modelid = 0;
            static int old_cpugpu = 0;
            if (taskid != old_taskid || (modelid % 3) != old_modelid || cpugpu != old_cpugpu)
            {
                // taskid or model or cpugpu changed
                delete g_yolov8;
                g_yolov8 = 0;
            }
            old_taskid = taskid;
            old_modelid = modelid % 3;
            old_cpugpu = cpugpu;

            ncnn::destroy_gpu_instance();

            if (use_turnip)
            {
                ncnn::create_gpu_instance("libvulkan_freedreno.so");
            }
            else if (use_gpu)
            {
                ncnn::create_gpu_instance();
            }

            if (!g_yolov8)
            {
                if (taskid == 0) g_yolov8 = new YOLOv8_det_coco;
                if (taskid == 1) g_yolov8 = new YOLOv8_det_oiv7;
                if (taskid == 2) g_yolov8 = new YOLOv8_seg;
                if (taskid == 3) g_yolov8 = new YOLOv8_pose;
                if (taskid == 4) g_yolov8 = new YOLOv8_cls;
                if (taskid == 5) g_yolov8 = new YOLOv8_obb;

                g_yolov8->load(mgr, parampath.c_str(), modelpath.c_str(), use_gpu || use_turnip);
            }
            int target_size = 320;
            if ((int)modelid >= 3)
                target_size = 480;
            if ((int)modelid >= 6)
                target_size = 640;
            g_yolov8->set_det_target_size(target_size);
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

JNIEXPORT jobject JNICALL Java_com_tencent_yolov8ncnn_YOLOv8Ncnn_getCurrentFrame(JNIEnv* env, jobject thiz)
{
    cv::Mat frame_copy;
    {
        ncnn::MutexLockGuard lock(g_latest_frame_lock);
        if (g_latest_frame_for_bitmap.empty()) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "getCurrentFrame: No frame available yet.");
            return nullptr;
        }
        frame_copy = g_latest_frame_for_bitmap.clone();
    }

    if (frame_copy.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "getCurrentFrame: Frame copy is empty after lock.");
        return nullptr;
    }

    return matToBitmap(env, frame_copy);
}

}