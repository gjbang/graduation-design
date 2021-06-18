//
// Created by axv on 2021/5/25.
//

#include "kputil.h"


namespace gj{

//    define global variables
    std::string output_path;
    int MAX_QUEUE = INTERS_NUM * 3;
    int INTERS_CNT = 0;
    char *tflite_path = "../models/openpose-singlenet.tflite";

//    define communicate queue and mutex between process
    std::queue<OutFrame> img_read_queue;
    std::mutex img_read_mutex;
    std::queue<OutFrame> img_write_queue;
    std::mutex img_write_mutex;
    std::mutex int_id_mutex;
//    tflite vars
//    tflite::ErrorReporter tf_error_reporter;


//    define thread pools
    typedef void (*ThreadMain)();
    std::map<std::string,ThreadMain >mapThreadMainManager={
            {"video_process",video_process},
//            {"mqtt",mqtt_send},
            {"kp_inference 1",kp_inference},
            {"kp_inference 2",kp_inference},
            {"kp_inference 3",kp_inference},
            {"kp_inference 4",kp_inference},
            {"kp_inference 5",kp_inference},
//            {"test1",test1},
//            {"test2",test2},
//            {"test3",test3}
    };


    Person::Person() {

    }
//

};