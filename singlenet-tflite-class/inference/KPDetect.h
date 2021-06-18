//
// Created by axv on 2021/5/25.
//

#ifndef SINGLENET_TFLITE_CLASS_KPDETECT_H
#define SINGLENET_TFLITE_CLASS_KPDETECT_H

#include "kputil.h"


namespace gj {
    using namespace tflite;
    using namespace std;
    using namespace cv;

    class KPDetect {
    public:
        int int_id;
        std::unique_ptr<FlatBufferModel> model;
        std::unique_ptr<Interpreter> interpreter;
        ops::builtin::BuiltinOpResolver op_resolver;

        KPDetect(int int_id);

        void print_debug_info_interpreter();

        void splitOutputToParts(TfLiteTensor *outputTensor, const cv::Size &targetSize, std::vector<cv::Mat> &parts, bool isPaf);
        void getKeyPoints(cv::Mat &probMap, double threshold, std::vector<KeyPoint> &keyPoints);

        void getPersonwiseKeypoints(const std::vector<std::vector<ValidPair>> &validPairs,
                                    const std::set<int> &invalidPairs,
                                    std::vector<std::vector<int>> &personwiseKeypoints);

        void getValidPairs(const std::vector<cv::Mat> &netOutputParts,
                           const std::vector<std::vector<KeyPoint>> &detectedKeypoints,
                           std::vector<std::vector<ValidPair>> &validPairs,
                           std::set<int> &invalidPairs);
        void populateInterpPoints(const cv::Point &a, const cv::Point &b, int numPoints, std::vector<cv::Point> &interpCoords);

        void getPersonPoints(vector<Person> &persons, std::vector<std::vector<int>>& personwise_kps,
                                       vector<KeyPoint>& kps_list);

        void recoverOriginalPoints(vector<Person>& can_persons,float scale,int offset, bool w_longer);
    };



};


#endif //SINGLENET_TFLITE_CLASS_KPDETECT_H
