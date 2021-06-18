//
// Created by axv on 2021/5/25.
//

#ifndef SINGLENET_TFLITE_CLASS_KPUTIL_H
#define SINGLENET_TFLITE_CLASS_KPUTIL_H

#include <string>
#include <queue>
#include <map>
#include <mutex>
#include <functional>
#include <iostream>
#include <cstring>
#include <cmath>
#include <vector>
#include <dirent.h>
#include <algorithm>
#include <thread>
#include <chrono>
#include <random>
#include <set>
#include <unordered_set>
#include <ctime>

// opencv headers
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/video/background_segm.hpp"

// tensorflow lite headers
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

// tensorflow lite macro header
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "./tensorflow/lite/version.h"
#include "tensorflow/lite/micro/examples/micro_speech/feature_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"

// use glog as logger
#include "glog/logging.h"


//    global setting - system
//#define LOG(severity) (std::cerr << (#severity) <<": ")
#define USE_VIDEO_FILE true
#define BIND_CPU  true
#define INTER_SHOW_VERBOSE  false
#define IMG_SHOW_VERBOSE  true
namespace gj {


//    const bool

//    global setting -key vars
    const int INPUT_SIZE = 224;
    const int INTERS_NUM = 5;
    extern int MAX_QUEUE;
    extern int INTERS_CNT;
    extern std::mutex int_id_mutex;


////////////////////////////////
    struct KeyPoint {
        KeyPoint(cv::Point point, float probability) {
            this->id = -1;
            this->point = point;
            this->probability = probability;
        }

        int id;
        cv::Point point;
        float probability;
    };

////////////////////////////////
    struct ValidPair {
        ValidPair(int aId, int bId, float score) {
            this->aId = aId;
            this->bId = bId;
            this->score = score;
        }

        int aId;
        int bId;
        float score;
    };

//    record single person's kps and calculate its centroid
    class Person{
        //        std::vector<struct KeyPoint> kps;

    public:
        Person();
        cv::Point centroid;
        cv::Point left_up;
        cv::Point right_down;
//        std::map<int,struct KeyPoint> kps;
        std::map<int,cv::Point> kps;
    };


    //    special data struct used to be transferred between threads or processes

    class OutFrame {
    public:
        long int frame_id = -1;
        uint8_t *input_data;
//        preprocess scale
        float scale;
//        preprocess offset
        int offset;
//        perprocess width > height
        bool w_longer;
//        left-up point when cropped from original image
        cv::Point left_up=cv::Point(-1,-1);
        cv::Point right_down;
//        std::vector<std::vector<struct KeyPoint>> detected_kps;
//        std::vector<struct KeyPoint> kps_list;
//        std::vector<std::vector<int>> personwise_kps;
        std::vector<Person> can_persons;

        bool operator<(const OutFrame &oframe) const {
            return oframe.frame_id < this->frame_id;
        }
    };
//    typedef struct out_frame *oframe;


//    define global variables
    extern char *tflite_path;
//    extern tflite::ErrorReporter tf_error_reporter;
    extern std::string output_path;


//    define communicate queue and mutex between process
    extern std::queue<OutFrame> img_read_queue;
    extern std::mutex img_read_mutex;
    extern std::queue<OutFrame> img_write_queue;
    extern std::mutex img_write_mutex;


//    define thread pools
    typedef void (*ThreadMain)();

    extern std::map<std::string, ThreadMain> mapThreadMainManager;

//    prcoess function name
    void video_process();

    void kp_inference();

    void mqtt_send();


    /* ALL FOR PAF and HEATMAP CAL*/
    const int n_points = 18;

    const std::string kp_mapping[] = {
            "Nose", "Neck",
            "R-Sho", "R-Elb", "R-Wr",
            "L-Sho", "L-Elb", "L-Wr",
            "R-Hip", "R-Knee", "R-Ank",
            "L-Hip", "L-Knee", "L-Ank",
            "R-Eye", "L-Eye", "R-Ear", "L-Ear"
    };


    const std::vector<std::pair<int, int>> map_idx = {
            {12, 13},
            {20, 21},
            {14, 15},
            {16, 17},
            {22, 23},
            {24, 25},
            {0, 1},
            {2, 3},
            {4, 5},
            {6, 7},
            {8, 9},
            {10, 11},
            {28, 29},
            {30, 31},
            {34, 35},
            {32, 33},
            {36, 37},
            {18, 19},
            {26, 27}
    };

    const std::vector<std::pair<int, int>> pose_pairs = {
            {1,  2},
            {1,  5},
            {2,  3},
            {3,  4},
            {5,  6},
            {6,  7},
            {1,  8},
            {8,  9},
            {9,  10},
            {1,  11},
            {11, 12},
            {12, 13},
            {1,  0},
            {0,  14},
            {14, 16},
            {0,  15},
            {15, 17},
            {2,  17},
            {5,  16}
    };


};


#endif //SINGLENET_TFLITE_CLASS_KPUTIL_H
