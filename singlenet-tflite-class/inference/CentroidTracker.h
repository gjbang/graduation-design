//
// Created by axv on 2021/5/28.
//

#ifndef SINGLENET_TFLITE_CLASS_CENTROIDTRACKER_H
#define SINGLENET_TFLITE_CLASS_CENTROIDTRACKER_H

#include "kputil.h"

namespace gj{
    using namespace std;
    using namespace tflite;
    using namespace cv;


    class CentroidTracker {

    public:
        long int cur_p_id = 0;

        int kp_num_thrs = 6;
        float cd_dis_thres = 2000.0f;
        int dis_frames_thres = 4;

        bool detect_fall=false;
        int fall_frame_cnt=0;

//        vector<Person> persons;
        map<long int,int> dis_frames;
        map<long int,Person> persons;
        vector<Person> candidate;
//        set<long int> fa

//
        void calCentroid();
        void updatePersons();
        void checkFall(Person &op, Person &np);
    };

    bool cmp_dis(const std::pair<long int,cv::Point>& a,const std::pair<long int,cv::Point>& b);
};





#endif //SINGLENET_TFLITE_CLASS_CENTROIDTRACKER_H
