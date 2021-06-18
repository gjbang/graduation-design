//
// Created by axv on 2021/5/25.
//

#include "KPDetect.h"

namespace gj {
    KPDetect::KPDetect(int int_id) {
        int_id = int_id;
        model = FlatBufferModel::BuildFromFile(tflite_path);
        if (!model) {
            LOG(ERROR) << "Failed to mmap model" << std::endl;
            exit(0);
        } else {
            LOG(INFO) << "Interpreter " << int_id << " has been created" << std::endl;
        }
        // init interpreter
        InterpreterBuilder(*model.get(), op_resolver)(&interpreter);
        DLOG(INFO) << "interpreter " << int_id << " has been inited" << std::endl;

        // init tensors
        interpreter->AllocateTensors();
        DLOG(INFO) << "tensor has been allocated" << std::endl;
    }

    void
    KPDetect::splitOutputToParts(TfLiteTensor *outputTensor, const cv::Size &targetSize, std::vector<cv::Mat> &parts,
                                 bool isPaf) {
//    get output tensors' information
        int nh = outputTensor->dims->data[1];
        int nw = outputTensor->dims->data[2];
        int nParts = outputTensor->dims->data[3];
        float32_t *p = (float32_t *) outputTensor->data.data;
//        cout << "nh: " << nh << "; nw:" << nw << "; nP:" << nParts << endl;

        int size[2] = {0, 0};
        size[0] = nh;
        size[1] = nw;

//        get data from output tensor and set channel is the number of parts
//        split original cv::Mat according to channels
        cv::Mat tensorData(2, size, CV_32FC(nParts), p);
        vector<cv::Mat> partsMats;
        cv::split(tensorData, partsMats);


        for (int i = 0; i < nParts; i++) {
            cv::Mat resizedPart;
            cv::resize(partsMats[i], resizedPart, targetSize, cv::INTER_CUBIC);
            parts.push_back(resizedPart);

#if IMG_SHOW_VERBOSE
//              save inter image -- heatmap and paf
                string name = isPaf ? "../output/paf-" : "heatmap-";
                name += to_string(i) + ".jpg";;

                cv::Mat noise = partsMats[i];
                cv::normalize(noise, noise, 0, 255, cv::NORM_MINMAX, -1);
                noise.convertTo(noise, CV_8U);

                cv::imwrite(name, noise);
#endif

        }
    }

// step1: elimate noise point
// step2: use 0-1 mat to blob the person's area
// step3: use max value in the person area as keypoints -> so just all kps in pics
    void KPDetect::getKeyPoints(cv::Mat &probMap, double threshold, std::vector<KeyPoint> &keyPoints) {
//    Gaussian filter -> eliminate noise -> kernel size:3 --> seems like scipy - mode:"reflect"
        cv::Mat smoothProbMap;
        cv::GaussianBlur(probMap, smoothProbMap, cv::Size(3, 3), 0, 0);

//    type: THRESH_BINARY : filter pixel with very low value -- delete
//    dst(x,y) = maxval if src(x,y) > thresh else 0
        cv::Mat maskedProbMap;
        cv::threshold(smoothProbMap, maskedProbMap, threshold, 255, cv::THRESH_BINARY);

//    adjust the type of mat -> uint8_t pic
        maskedProbMap.convertTo(maskedProbMap, CV_8U, 1);

//    find edge of 0-1 image;
//    mode:"RETR_TREE" - represent edge using tree format, inner edge will be child of outer edge
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(maskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);


        for (int i = 0; i < contours.size(); ++i) {
//        init a zeros mat as inited mat
            cv::Mat blobMask = cv::Mat::zeros(smoothProbMap.rows, smoothProbMap.cols, smoothProbMap.type());
//        padding convex with 1 using edge points detected before
            cv::fillConvexPoly(blobMask, contours[i], cv::Scalar(1));

            double maxVal;
            cv::Point maxLoc;
//        find the min or max value in a mat, if param not needed, just set 0 and NULL
//        in fact, use edge detect to blob the person area, then get the max value of heatmap in this area
            cv::minMaxLoc(smoothProbMap.mul(blobMask), 0, &maxVal, 0, &maxLoc);
//        satisfy the keypoints in cur heatmap
            keyPoints.push_back(KeyPoint(maxLoc, probMap.at<float>(maxLoc.y, maxLoc.x)));
        }
    }

    // Fundamental purpose: for calculating the line integral
// Pratical Method: calculate the interpoints between two points by average sampling
// Concrete Equation : p(u) = (1-u) * d_{j1} + u * d_{j2}
// Return Value : vectors of delta ( p(u) - a )
    void KPDetect::populateInterpPoints(const cv::Point &a, const cv::Point &b, int numPoints,
                                        std::vector<cv::Point> &interpCoords) {
//    calculate delta x and delta y
        float xStep = ((float) (b.x - a.x)) / (float) (numPoints - 1);
        float yStep = ((float) (b.y - a.y)) / (float) (numPoints - 1);

//    the start point
        interpCoords.push_back(a);

        for (int i = 1; i < numPoints - 1; ++i) {
            interpCoords.push_back(cv::Point(a.x + xStep * i, a.y + yStep * i));
        }
//    the end point
        interpCoords.push_back(b);
    }


    void KPDetect::getValidPairs(const std::vector<cv::Mat> &netOutputParts,
                                 const std::vector<std::vector<KeyPoint>> &detectedKeypoints,
                                 std::vector<std::vector<ValidPair>> &validPairs,
                                 std::set<int> &invalidPairs) {

        int nInterpSamples = 14;
        float pafScoreTh = 0.1;
        float confTh = 0.8;

//    for limbs -> mapIdx can change into lambda (x-19)
        for (int k = 0; k < map_idx.size(); ++k) {

            //A->B constitute a limb
//            cv::Mat pafA = netOutputParts[map_idx[k].first -19];
//            cv::Mat pafB = netOutputParts[map_idx[k].second -19];

            cv::Mat pafA = netOutputParts[map_idx[k].first];
            cv::Mat pafB = netOutputParts[map_idx[k].second];


            //Find the keypoints for the first and second limb -> may be more than one kp
            const std::vector<KeyPoint> &candA = detectedKeypoints[pose_pairs[k].first];
            const std::vector<KeyPoint> &candB = detectedKeypoints[pose_pairs[k].second];

            int nA = candA.size();
            int nB = candB.size();

            /*
              # If keypoints for the joint-pair is detected
              # check every joint in candA with every joint in candB
              # Calculate the distance vector between the two joints
              # Find the PAF values at a set of interpolated points between the joints
              # Use the above formula to compute a score to mark the connection valid
             */

            if (nA != 0 && nB != 0) {
                std::vector<ValidPair> localValidPairs;

                for (int i = 0; i < nA; ++i) {
                    int maxJ = -1;
                    float maxScore = -1;
                    bool found = false;

                    for (int j = 0; j < nB; ++j) {
//                    get |x| and  |y|
                        std::pair<float, float> distance(candB[j].point.x - candA[i].point.x,
                                                         candB[j].point.y - candA[i].point.y);
//                    get sqrt(x^2 + y^2)
                        float norm = std::sqrt(distance.first * distance.first + distance.second * distance.second);

//                    if norm == 0, then two body parts may overlaps, so this connection need to be skipped
                        if (norm == 0) {
                            continue;
                        }

//                    get vec.x and vec.y -> normalize
                        distance.first /= norm;
                        distance.second /= norm;

                        //Find p(u)
//                    get the set sample points between 2 body parts
                        std::vector<cv::Point> interpCoords;
                        populateInterpPoints(candA[i].point, candB[j].point, nInterpSamples, interpCoords);

                        //Find L(p(u))
//                    after having gotten coordinates of interpoints, read intensity val in paf map at this point
//                    first: y, second x
                        std::vector<std::pair<float, float>> pafInterp;
                        for (int l = 0; l < interpCoords.size(); ++l) {
//                        pafA and pafB is cv::Mat, so 'at' method needs (row,col) -> which in fact is (y,x) in opencv
//                        pafA -- dx component ; pafB -- dy component -> which give a paf heatmap direction
//                        this direction can help find the real connection between some candidates
                            pafInterp.push_back(
                                    std::pair<float, float>(
                                            pafA.at<float>(interpCoords[l].y, interpCoords[l].x),
                                            pafB.at<float>(interpCoords[l].y, interpCoords[l].x)
                                    ));
                        }

//                    calculate the score for the connection weighted by distance between body parts
//                    L_{c,k}(p)
                        std::vector<float> pafScores;
//                    L_{c}(p) * n_{c}(P) -> not the final average value
                        float sumOfPafScores = 0;
//                    number of nonzero vector -> k people's same parts
                        int numOverTh = 0;

//                    sum all L{c,k} (p)
                        for (int l = 0; l < pafInterp.size(); ++l) {
//                        s1: v * (p(u) - x) -> dot product -> projection on one vector
//                        condition1 : 0 <= s1 <= norm
//                        v * (p(u)-x) = |v| * |p(u)-x| * cos, |v|=1, cos<1, |p(u)-x| is small, so c1 can hold easily
//                        condition2 : s2=v_{T} * (p(u) - x) -> x*y + y*x

//                        this is just v
                            float score = pafInterp[l].first * distance.first + pafInterp[l].second * distance.second;
//                        total score -> sum of L_{c,k}(p)
                            sumOfPafScores += score;

//                        condition2:  s2<= threshold
//                        satisfy number of points with score intensity in inter points
                            if (score > pafScoreTh) {
                                ++numOverTh;
                            }

//                        just save all p(u)'s L_{c,k}(p(u))
                            pafScores.push_back(score);
                        }

//                   E = ~L(p(u)) = ~ sum(L_{c,k}(p(u))) / n_{c}(p(u)) du
                        float avgPafScore = sumOfPafScores / ((float) pafInterp.size());

//                    connection candidate condition 1:
//                    ratio of p(u) with score > th-pafScoreTh should be more than confTH
//                    number of midpoints with intensity above the threshold shouldn more than 80% of all midpoints
                        if (((float) numOverTh) / ((float) nInterpSamples) > confTh) {
//                        s1: max E for get multi people's connection
//                        do: record the  max score and record the corresponding candidate id of can B
                            if (avgPafScore > maxScore) {
                                maxJ = j;
                                maxScore = avgPafScore;
                                found = true;
                            }
                        }

                    }/* j */

                    if (found) {
                        localValidPairs.push_back(ValidPair(candA[i].id, candB[maxJ].id, maxScore));
                    }

                }/* i */
//            get valid connection
                validPairs.push_back(localValidPairs);

            } else {
//            insert an empty vector -> not get effective connection
                invalidPairs.insert(k);
                validPairs.push_back(std::vector<ValidPair>());
            }
        }/* k */
    }


    // after joined all the keypoints into pairs
// can assemble the pairs that share the same part detection candidates into fully-body pose of multi people
    void KPDetect::getPersonwiseKeypoints(const std::vector<std::vector<ValidPair>> &validPairs,
                                          const std::set<int> &invalidPairs,
                                          std::vector<std::vector<int>> &personwiseKeypoints) {
//    for all parts need to be parsed
        for (int k = 0; k < map_idx.size(); ++k) {
//        skip invalid pairs for some specific parts
            if (invalidPairs.find(k) != invalidPairs.end()) {
                continue;
            }
//        temp vars
            const std::vector<ValidPair> &localValidPairs(validPairs[k]);

//        defined in header file, which is the connection parts' indices
//        index A: the start part index; index B: the end part index
            int indexA(pose_pairs[k].first);
            int indexB(pose_pairs[k].second);

//        for one parts -> connect pairs
            for (int i = 0; i < localValidPairs.size(); ++i) {
                bool found = false;
                int personIdx = -1;

//            j means person id-> at the first, there is no person in list
                for (int j = 0; !found && j < personwiseKeypoints.size(); ++j) {
//                for person j, check if part A is already in this person pairs list
                    if (indexA < personwiseKeypoints[j].size() &&
                        personwiseKeypoints[j][indexA] == localValidPairs[i].aId) {
                        personIdx = j;
//                    1. helps to quit this loop; 2.helps to push part B into list
                        found = true;
                    }
                }/* j */
//            if part A belongs to the person, then part B in valid pair also belongs to this person
                if (found) {
                    personwiseKeypoints[personIdx].at(indexB) = localValidPairs[i].bId;
                }
//            if current part A is not present in any of list, then it must belong to a new person not in cur list
//            so create this person list, and add part A into it
                else if (k < 17) {
//                create one person's connection pairs vector
                    std::vector<int> lpkp(std::vector<int>(18, -1));

                    lpkp.at(indexA) = localValidPairs[i].aId;
                    lpkp.at(indexB) = localValidPairs[i].bId;

                    personwiseKeypoints.push_back(lpkp);
                }

            }/* i */
        }/* k */
    }

    void KPDetect::print_debug_info_interpreter() {
        // there is one input tensor -> uint8
        int in_size = interpreter->inputs().size();
        cout << "[DEBUG] number of input tensors in interpreter is: " << in_size << endl;
        for (int i = 0; i < in_size; i++) {
            int cur_dim_size = interpreter->input_tensor(i)->dims->size;
            DLOG(INFO) << "index: " << i << "; name: " << interpreter->GetInputName(i) << "; dims cnt: " << cur_dim_size
                       << "; data type: " << interpreter->input_tensor(i)->type << endl;
            for (int d = 0; d < cur_dim_size; ++d) {
                DLOG(INFO) << "dim index: " << d << "; dimensiona: " << interpreter->input_tensor(i)->dims->data[d]
                           << endl;
            }
        }

        // there are four output tensor -> float32
        int out_size = interpreter->outputs().size();
        cout << "[DEBUG] number of output tensors in interpreter is: " << out_size << endl;
        for (int i = 0; i < out_size; i++) {
            int cur_dim_size = interpreter->output_tensor(i)->dims->size;
            DLOG(INFO) << "index: " << i << "; name: " << interpreter->GetOutputName(i) << "; dims cnt: "
                       << cur_dim_size
                       << "; data type: " << interpreter->output_tensor(i)->type << endl;
            for (int d = 0; d < cur_dim_size; ++d) {
                DLOG(INFO) << "dim index: " << d << "; dimensiona: " << interpreter->output_tensor(i)->dims->data[d]
                           << endl;
            }
        }

        //
        int nodes_size = interpreter->nodes_size();
        DLOG(INFO) << " number of nodes tensors in interpreter is: " << nodes_size << endl;

        int tensors_size = interpreter->tensors_size();
        DLOG(INFO) << " number of all tensors in interpreter is: " << tensors_size << endl;


#if INTER_SHOW_VERBOSE
            LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
            LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
            LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
            LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

            int t_size = interpreter->tensors_size();
            for (int i = 0; i < t_size; i++) {
                if (interpreter->tensor(i)->name)
                    LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                              << interpreter->tensor(i)->bytes << ", "
                              << interpreter->tensor(i)->type << ", "
                              << interpreter->tensor(i)->params.scale << ", "
                              << interpreter->tensor(i)->params.zero_point << "\n";
            }
#endif
    }

    void KPDetect::getPersonPoints(vector<Person> &persons, std::vector<std::vector<int>>& personwise_kps,
                                   vector<KeyPoint>& kps_list) {
        //        vector<Person> persons;
        persons.clear();
        for(int p=0;p<personwise_kps.size();++p){
            Person cp;
//            use for getting bounding box -> for falling detection
            cv::Point lu=cv::Point(300,300);
            cv::Point rd=cv::Point(-1,-1);
            for(int kp = 0;kp<n_points -1 ;kp++){
                int indA = personwise_kps[p][pose_pairs[kp].first];
                int indB = personwise_kps[p][pose_pairs[kp].second];
                if (indA != -1 && cp.kps.find(indA)==cp.kps.end()){
                    cv::Point ckp = kps_list[indA].point;
                    cp.kps.insert(std::pair<int,cv::Point> (indA,ckp));
                    lu.x=lu.x>ckp.x?ckp.x:lu.x;
                    lu.y=lu.y>ckp.y?ckp.y:lu.y;
                    rd.x=rd.x<ckp.x?ckp.x:rd.x;
                    rd.y=rd.y<ckp.y?ckp.y:rd.y;
                }
                if (indB != -1 && cp.kps.find(indB)==cp.kps.end()){
                    cv::Point ckp = kps_list[indB].point;
                    cp.kps.insert(std::pair<int,cv::Point> (indB,ckp));
                    lu.x=lu.x>ckp.x?ckp.x:lu.x;
                    lu.y=lu.y>ckp.y?ckp.y:lu.y;
                    rd.x=rd.x<ckp.x?ckp.x:rd.x;
                    rd.y=rd.y<ckp.y?ckp.y:rd.y;
                }
            }
//            set left-up and right-down points -> which is a bounding box
            cp.left_up=lu;
            cp.right_down=rd;
            persons.push_back(cp);
        }
        DLOG(INFO) <<" get person points, persons num:"<<persons.size()<<std::endl;

    }

    void KPDetect::recoverOriginalPoints(vector<Person> &can_persons, float scale, int offset,bool w_longer) {
        if(w_longer){
            for(auto&p:can_persons){
                p.centroid.y-=offset;
                p.left_up.y-=offset;
                p.right_down.y-=offset;
                for(auto& kp:p.kps){
                    kp.second.y-=offset;
                }
            }
        }else{
            for(auto&p:can_persons){
                p.centroid.x-=offset;
                p.left_up.x-=offset;
                p.right_down.x-=offset;
                for(auto& kp:p.kps){
                    kp.second.x-=offset;
                }
            }
        }
        for(auto &p:can_persons){
            p.centroid /= scale;
            p.left_up /= scale;
            p.right_down /= scale;
            for(auto&kp:p.kps){
                kp.second /= scale;
            }
        }
    }


    /* KP detect main loop */
    void kp_inference() {
        int cur_int_id = 0;
        {
            std::lock_guard<std::mutex> lock(int_id_mutex);
            cur_int_id = INTERS_CNT++;
        }

        KPDetect kpDetect = KPDetect(cur_int_id);
        if (cur_int_id == 0) {
            kpDetect.print_debug_info_interpreter();
        }

//        iframe cur_iframe;
        OutFrame cur_iframe;
        TfLiteTensor *input_tensor = kpDetect.interpreter->input_tensor(0);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        while (true) {

            /*This part is for getting input data*/
            {
                bool read_queue=false;
                {//                get the img_read_queue
                    std::lock_guard<std::mutex> lock(img_read_mutex);
                    read_queue = !img_read_queue.empty();
                }
                if (read_queue) {
                    std::lock_guard<std::mutex> lock(img_read_mutex);
                    cur_iframe = img_read_queue.front();
                    img_read_queue.pop();
                    DLOG(INFO) << " get image: " << cur_iframe.frame_id << std::endl;
                } else {
                    DLOG(INFO) <<"[INT"<<cur_int_id<<"] is sleeping";
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                    continue;
                }


            }
//
            /*do inference*/
//            set input tensor <- from a uint8_t data ptr
            input_tensor->data.uint8 = cur_iframe.input_data;
            std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
            if (kTfLiteOk == kpDetect.interpreter->Invoke()) {
                DLOG(INFO) <<  "interpreter inference normally" << std::endl;
            } else {
                LOG(ERROR) << "interpreter cannot invoke\n";
            }
            std::chrono::time_point<std::chrono::system_clock> finishTP = std::chrono::system_clock::now();
            LOG(INFO) << "[INT"<<cur_int_id<<"] frame: "<<cur_iframe.frame_id<<"; Time Taken in forward pass = "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP - startTP).count() << " ms"
                      << std::endl;


//            /*calculate paf and heatmap*/
//        split 19 parts vector into different cv::Mat
            TfLiteTensor *heatmap = kpDetect.interpreter->output_tensor(3);
            vector<cv::Mat> output_parts;
            kpDetect.splitOutputToParts(heatmap, cv::Size(224, 224), output_parts, false);
            DLOG(INFO) << "heatmap tensor split parts finish" << endl;
//
//        split 38 pafs vector into different cv::Mat
            TfLiteTensor *paf = kpDetect.interpreter->output_tensor(2);
            vector<cv::Mat> output_pafs;
            kpDetect.splitOutputToParts(paf, cv::Size(224, 224), output_pafs, true);
            DLOG(INFO) << "paf tensor split has been finished" << endl;


//        all detected keypoints -> dim1: parts ; dim2: candidate keypoint for specific part
            vector<vector<KeyPoint>> detected_kps;
            vector<KeyPoint> kps_list;
//            this id is used in kps_list
            int cur_kp_id = 0;

            for (int kp_id = 0; kp_id < n_points; ++kp_id) {
//            one single kind of kp
                vector<KeyPoint> kps;

                kpDetect.getKeyPoints(output_parts[kp_id], 1e-1, kps);

//            set kps id using cur id -> allocate kinds -> not kp-mark id, just the orider
                for (int j = 0; j < kps.size(); ++j, ++cur_kp_id) {
                    kps[j].id = cur_kp_id;
                }

//            save kps according kinds
                detected_kps.push_back(kps);
//            save kps in a linear order
                kps_list.insert(kps_list.end(), kps.begin(), kps.end());
            }

            /*get valid pairs*/
            std::vector<std::vector<ValidPair>> valid_pairs;
            std::set<int> invalid_pairs;
            kpDetect.getValidPairs(output_pafs, detected_kps, valid_pairs, invalid_pairs);
            DLOG(INFO) << " Get valid pairs ok";

            /*get personwise kps -- connect kps*/
            std::vector<std::vector<int>> personwise_kps;
            kpDetect.getPersonwiseKeypoints(valid_pairs, invalid_pairs, personwise_kps);
            DLOG(INFO) <<  " Get personwise keypoints ok";

            /*get person with kps*/
            std::vector<Person> can_persons;
            kpDetect.getPersonPoints(can_persons,personwise_kps,kps_list);
            DLOG(INFO) <<" Get each person with concreate kps";

            /*recover points to original coordinates*/
//            kpDetect.recoverOriginalPoints(can_persons,cur_iframe.scale,cur_iframe.offset,cur_iframe.w_longer);


            /*past data for post-processing*/
            while (true) {
//                only get mutex when checking write queue and enqueue
                {
                    std::lock_guard<std::mutex> lock(img_write_mutex);
//                    check if write has been filled
                    if (img_write_queue.size() < MAX_QUEUE) {
//                        cur_iframe.detected_kps = detected_kps;
//                        cur_iframe.kps_list=kps_list;
//                        cur_iframe.personwise_kps=personwise_kps;
                        cur_iframe.can_persons=can_persons;
                        img_write_queue.push(cur_iframe);
                        DLOG(INFO) << "[INT " << cur_int_id << "]" << cur_iframe.frame_id << " has been processed"
                                   << "; in_ptr has been released" << std::endl;
                        break;
                    }
                }
//                if cannot enqueue, wait next loop
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
            }

        }
    }

};

