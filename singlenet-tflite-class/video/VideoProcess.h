//
// Created by axv on 2021/5/25.
//

#ifndef SINGLENET_TFLITE_CLASS_VIDEOPROCESS_H
#define SINGLENET_TFLITE_CLASS_VIDEOPROCESS_H

#include "kputil.h"
#include "CentroidTracker.h"
#include "smqtt.h"
#include "base64Class.h"

namespace gj {
    class VideoProcess {

    private:

    public:
//        video stream
        cv::VideoCapture capture;
        cv::VideoWriter writer;
        long int frame_id=0;
        priority_queue<OutFrame> out_queue;

//        inpur data buffer
        std::queue<uint8_t*> buffer_ptr;

//        prepare for read video
        std::string video_path;
        bool is_read_file;
        long int video_len;
        std::map<long int,cv::Mat> video_out;
        std::vector<cv::Scalar> colors;

//        postprocess - calculate centroid- check fall
        CentroidTracker ct;


//        constructor function, default for camera, the other for video file
        VideoProcess();
        VideoProcess(std::string video_path);
//        init buffer
        bool init_buffer();
        void populateColorPalette();
        void preprocess_image(cv::Mat& input_img, OutFrame& cur_frame);
        void drawSingleFrame(OutFrame& out_data);
        void recoverOriginalPoints(vector<Person>& can_persons,float scale,int offset, bool w_longer,cv::Point&left_up);
        bool checkFrontGround(cv::Mat& front_mask,OutFrame& cur_frame);
    };


    bool cmp_frame(OutFrame&a,OutFrame&b);

};


#endif //SINGLENET_TFLITE_CLASS_VIDEOPROCESS_H
