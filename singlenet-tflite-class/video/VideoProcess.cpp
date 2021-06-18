//
// Created by axv on 2021/5/25.
//

#include "VideoProcess.h"

namespace gj {


    bool cmp_frame(OutFrame &a, OutFrame &b) {
        return a.frame_id < b.frame_id;
    }

    VideoProcess::VideoProcess() {

//        get the camera stream and init setting
        capture.open(-1);
        if (!capture.isOpened()) {
            capture.open(0);
            LOG(WARNING) << "camera 0 cannot be opened!" << std::endl;
        }
        if (!capture.isOpened()) {
            capture.open(1);
            LOG(ERROR) << " all camera cannot be opened!" << std::endl;
            exit(0);
        }

        capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
        capture.set(cv::CAP_PROP_FPS, 10);
        capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, 960);
        is_read_file = false;
        DLOG(INFO) << " video capture from camera has been inited" << std::endl;

        init_buffer();
        populateColorPalette();

    }

    VideoProcess::VideoProcess(std::string video_path) {
//        get video stream and init setting
        video_path = video_path;
        capture.open(video_path);
        if (!capture.isOpened()) {
            LOG(ERROR) << " video cannot be opened!" << std::endl;
            exit(0);
        }
        capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
        is_read_file = true;
        video_len = capture.get(cv::CAP_PROP_FRAME_COUNT);
        int fps = capture.get(cv::CAP_PROP_FPS);
        int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        DLOG(INFO) << " video from file has been inited, length: " << video_len << std::endl;

        int fourcc = cv::VideoWriter::fourcc('H', '2', '6', '4');
//        writer.open("../output/out.mp4", fourcc, fps, cv::Size(224, 224));
        writer = cv::VideoWriter("../output/out.mp4", fourcc, fps, cv::Size(width, height));


        init_buffer();
        populateColorPalette();
    }


    bool VideoProcess::init_buffer() {
        for (int i = 0; i < MAX_QUEUE * 2; ++i) {
            uint8_t *buffer = (uint8_t *) malloc(sizeof(uint8_t) * INPUT_SIZE * INPUT_SIZE * 3);
            buffer_ptr.push(buffer);
            if (buffer == nullptr) {
                LOG(ERROR) << " video buffer init fails" << std::endl;
//                return false;
                exit(0);
            }
        }

        return true;
    }

    void VideoProcess::populateColorPalette() {
//    come out a random num as seed
        std::random_device rd;
//    random num producer -> Mersenne Twister algorithm
        std::mt19937 gen(rd());
//    set random num producer in a specified range
        std::uniform_int_distribution<> dis1(64, 200);
        std::uniform_int_distribution<> dis2(100, 255);
        std::uniform_int_distribution<> dis3(100, 255);
//    save different color for different kp
        for (int i = 0; i < n_points; ++i) {
            colors.push_back(cv::Scalar(dis1(gen), dis2(gen), dis3(gen)));
        }
    }

    void VideoProcess::preprocess_image(cv::Mat &input_img, OutFrame &cur_frame) {
        uint8_t *input_data = cur_frame.input_data;
        long int frame_id = cur_frame.frame_id;
        video_out.insert(std::pair<long int, cv::Mat>(frame_id, input_img));

        cv::Point lu = cur_frame.left_up;
        cv::Point ru = cur_frame.right_down;
        if (lu.x >= 0) {
//            cv::Mat temp;
//            input_img.copyTo(temp);
            cv::Rect rect = cv::Rect(lu.x, lu.y, ru.x - lu.x, ru.y - lu.y);
//            temp = temp(rect);
            input_img=input_img(rect);
//            cv::imwrite("../output/partcrop/part" + to_string(cur_frame.frame_id) + ".jpg", temp);
        }


//      do resize, adjust the shorter edge into 224
        int img_w = input_img.cols;
        int img_h = input_img.rows;
//        DLOG(INFO) <<"w:"<<img_w <<"; h:"<<img_h<<std::endl;
        float scale = img_w > img_h ? INPUT_SIZE * 1.0f / img_w : INPUT_SIZE * 1.0f / img_h;
        cv::resize(input_img, input_img, cv::Size(0, 0), scale, scale);
        cur_frame.scale = scale;

//    store sub-stage iamge
        input_img.convertTo(input_img, CV_32FC3);
#if IMG_SHOW_VERBOSE
            cv::imwrite("../output/resize/resize-" + std::to_string(frame_id) + ".jpg", input_img);
#endif

//    pad value of (128,128,128) -> make image (224,224)
        int r_img_h = input_img.rows;
        int r_img_w = input_img.cols;
        int offset = 0;
        if (r_img_w > r_img_h) {
            offset = ceil((r_img_w - r_img_h) / 2);
            int hoff=offset;
            if(offset*2 + r_img_h<224){
                hoff=224-r_img_h-offset;
            }
            cv::copyMakeBorder(input_img, input_img, offset, hoff, 0, 0, cv::BORDER_CONSTANT,
                               cv::Scalar(128, 128, 128));
//            cv::Rect rect(offset,0,INPUT_SIZE,INPUT_SIZE);
//            input_img = input_img(rect);
            cur_frame.w_longer = true;
            cur_frame.offset = offset;
        } else {

            offset = ceil((r_img_h - r_img_w) / 2);
            int roff=offset;
            if(roff*2+r_img_w<224){
                roff=224-r_img_w-offset;
            }
            cv::copyMakeBorder(input_img, input_img, 0, 0, offset, roff, cv::BORDER_CONSTANT,
                               cv::Scalar(128, 128, 128));
//            cv::Rect rect(0,offset,INPUT_SIZE,INPUT_SIZE);
//            input_img = input_img(rect);
            cur_frame.w_longer = false;
            cur_frame.offset = offset;
        }

//        store crop-pad image
#if IMG_SHOW_VERBOSE
        DLOG(INFO) << " new image (h,w): (" << input_img.rows << ", " << input_img.cols << ")"
                   << "; offset: "
                   << offset << std::endl;
            cv::imwrite("../output/padcrop/pad_or_crop-" + std::to_string(frame_id) + ".jpg", input_img);
//            seems if directly convert original input_img, will cause ptr copy exit code 139 -> segmentation fault
        cv::Mat save_img;
        input_img.convertTo(save_img, CV_8U);
//            video_out.insert(std::pair<long int,cv::Mat>(frame_id,save_img));
#endif

        //    copy image data into input tensor's data ptr
        float *img_data = (float *) input_img.data;
        float mean[3] = {0, 0, 0};

        for (int h = 0; h < INPUT_SIZE; h++) {
            for (int w = 0; w < INPUT_SIZE; w++) {
                for (int c = 0; c < 3; c++) {
//                224 * 224 * 3
//                    input_data[c + h * INPUT_SIZE * 3 + w * 3] = (uint8_t) (*img_data - mean[c]);
                    *(input_data + c + h * INPUT_SIZE * 3 + w * 3) = (uint8_t) (*img_data - mean[c]);
                    img_data++;
                }

            }
        }
        DLOG(INFO) << frame_id << " has been set input ptr" << std::endl;
    }

    void VideoProcess::drawSingleFrame(OutFrame &out_data) {

        /*concrete draw*/
        cv::Scalar color_text(12, 255, 200);
        cv::Scalar color_rect(0, 255, 0);
        cv::Scalar color_fall(255, 0, 0);
//        DLOG(INFO)<<"output size:"<<video_out[out_data.frame_id].rows<<","<<video_out[out_data.frame_id].cols;
        for (auto &p:ct.persons) {

            std::map<int, cv::Point> kps = p.second.kps;

            for (auto &kp:kps) {
//                DLOG(INFO)<<"new : ( "<<kp.second.x<<", "<<kp.second.y<<")";
                cv::circle(video_out[out_data.frame_id], kp.second, 1, colors[kp.first], -1, cv::LINE_AA);
            }

            for (int i = 0; i < n_points - 1; i++) {
                const std::pair<int, int> &posePair = pose_pairs[i];
                if (kps.find(posePair.first) != kps.end() && kps.find(posePair.second) != kps.end()) {
                    cv::line(video_out[out_data.frame_id], kps.find(posePair.first)->second,
                             kps.find(posePair.second)->second, colors[i], 1, cv::LINE_AA);
                }
            }

            cv::putText(video_out[out_data.frame_id], to_string(p.first), p.second.centroid, cv::FONT_HERSHEY_COMPLEX,
                        0.5, color_text, 1);
            cv::rectangle(video_out[out_data.frame_id], p.second.left_up, p.second.right_down, color_rect, 1, LINE_8);


        }
        if (ct.detect_fall) {
            cv::putText(video_out[out_data.frame_id], "Fall", cv::Point(100, 200), cv::FONT_HERSHEY_COMPLEX, 0.5,
                        color_fall, 1);
        }
        DLOG(INFO) << out_data.frame_id << " has been marked" << std::endl;
    }

    void VideoProcess::recoverOriginalPoints(vector<Person> &can_persons, float scale, int offset, bool w_longer,cv::Point&left_up) {
        if (w_longer) {
            for (auto &p:can_persons) {
                p.centroid.y -= offset;
                p.left_up.y -= offset;
                p.right_down.y -= offset;
                for (auto &kp:p.kps) {
                    kp.second.y -= offset;
                }
            }
        } else {
            for (auto &p:can_persons) {
                p.centroid.x -= offset;
                p.left_up.x -= offset;
                p.right_down.x -= offset;
                for (auto &kp:p.kps) {
                    kp.second.x -= offset;
                }
            }
        }
        for (auto &p:can_persons) {
            p.centroid /= scale;
            p.left_up /= scale;
            p.right_down /= scale;
            for (auto &kp:p.kps) {
                kp.second /= scale;
            }
        }

        if(left_up.x>=0){
            for(auto &p:can_persons){
                p.centroid +=left_up;
                p.left_up+=left_up;
                p.right_down+=left_up;
                for(auto &kp:p.kps){
                    kp.second +=left_up;
                }
            }
        }

    }

    bool VideoProcess::checkFrontGround(cv::Mat &front_mask, OutFrame &cur_frame) {
//        make 0-1 image
//        cv::threshold(front_mask, front_mask, 200, 255, THRESH_BINARY);
//        get contours
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(front_mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        double area_thres = 120;
        double len_thres = 30;
        cv::Point ru = cv::Point(0, 0);
        cv::Point lu = cv::Point(front_mask.cols, front_mask.rows);
        bool has_move = false;
        for (size_t ci = 0; ci < contours.size(); ++ci) {
//            if (contourArea(contours[ci]) > area_thres) {
            if (cv::arcLength(contours[ci], true) > len_thres) {
                cv::Rect temp_rect = cv::boundingRect(cv::Mat(contours[ci]));
                lu.x = lu.x > temp_rect.x ? temp_rect.x : lu.x;
                lu.y = lu.y > temp_rect.y ? temp_rect.y : lu.y;
                ru.x = max(ru.x, temp_rect.x + temp_rect.width);
                ru.y = max(ru.y, temp_rect.y + temp_rect.height);
            }
        }

        if (ru.x - lu.x > 100 && ru.y - lu.y > 100) {
            cur_frame.left_up = lu;
            cur_frame.right_down = ru;
            has_move = true;
        }

        return has_move;
    }


    void video_process() {
        base64Class base64;
        bool is_falling=false;

        time_t now= time(0);

        sample_mem_persistence persist;
        mqtt::client m_client(SERVER_ADDRESS,CLIENT_ID,&persist);
        user_callback cb;
        m_client.set_callback(cb);

        mqtt::connect_options connOpts;
        connOpts.set_keep_alive_interval(20);
        connOpts.set_clean_session(true);
        connOpts.set_user_name("axv");
        connOpts.set_password("123456");
        DLOG(INFO)<<"mqtt client init ok";

        try {
            m_client.connect(connOpts);
            LOG(INFO)<<"mqtt connect ok";
        }
        catch (const mqtt::persistence_exception& exc) {
            std::cerr << "Persistence Error: " << exc.what() << " ["
                      << exc.get_reason_code() << "]" << std::endl;
//            return 1;
            exit(0);
        }
        catch (const mqtt::exception& exc) {
            std::cerr << exc.what() << std::endl;
//            return 1;
            exit(0);
        }

        bool has_cap = false;
        cv::Mat rgb_image, last_image, front_mask;
        long int last_frame_id = -1;
        bool need_img = false;
        Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
        cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2), cv::Point(-1, -1));
//        Ptr<BackgroundSubtractorGMG> bgsubtractor=createBackgroundSubtractorGMG();

        bgsubtractor->setVarThreshold(10);
        bgsubtractor->setHistory(40);
//        bgsubtractor->setVarThresholdGen(10);
        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();

#ifdef USE_VIDEO_FILE
        VideoProcess vp = VideoProcess("../resources/fall-10-cam0-rgb.avi");
#else
        VideoProcess vp = VideoProcess();
#endif

//        main_loop
        while (true) {
            /*This part push data into img_read queue*/
            {
                {//            check img read queue if need add new data
                    DLOG(INFO)<<"video is waiting for read mutex";
                    std::lock_guard<std::mutex> lock(img_read_mutex);
                    need_img = img_read_queue.size() < MAX_QUEUE;

                }
                DLOG(INFO)<<"video need read new image";
//            add new input data point into queue
                if (need_img) {
                    LOG(INFO) << "frame id:" << vp.frame_id << "; current read queue:" << img_read_queue.size();
                    /* read original image data*/
                    has_cap = vp.capture.read(rgb_image);
//                cannot capture, exception op
                    if (!has_cap) {
                        if (vp.is_read_file) {
                            LOG(INFO) << " Video file has been read all " << std::endl;
                            vp.writer.release();
                            exit(0);
//                            break;
                        } else {
                            LOG(ERROR) << " Video cap cannot read image from camera" << std::endl;
                            exit(0);
                        }
                    }

                    /*do single crop*/
                    /*get frontground*/
                    bgsubtractor->apply(rgb_image, front_mask);
//                    decrease noise point
                    morphologyEx(front_mask, front_mask, cv::MORPH_OPEN, kernel);
//                    cv::imwrite("../output/front/front" + to_string(vp.frame_id) + ".jpg", front_mask);
                    OutFrame cur_frame;
//                    if (vp.checkFrontGround(front_mask, cur_frame)) {
#if IMG_SHOW_VERBOSE
                    cv::imwrite("../output/front/front"+ to_string(vp.frame_id)+".jpg",front_mask);
#endif

//                init input data buffer
                        uint8_t *cur_buffer_p;
                        while (true) {
                            if (!vp.buffer_ptr.empty()) {
//                        DLOG(INFO) << "[VIDEO] cur buffer size:" << vp.buffer_ptr.size() << std::endl;
                                LOG(INFO) << "frame id:" << vp.frame_id << "; current buffer queue:"
                                          << vp.buffer_ptr.size();
                                cur_buffer_p = vp.buffer_ptr.front();
                                vp.buffer_ptr.pop();
                                break;
                            }
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        }

//                pack input data into one frame

                        cur_frame.frame_id = vp.frame_id;
                        cur_frame.input_data = cur_buffer_p;
//                preprocess image data and copy dara into
                        vp.preprocess_image(rgb_image, cur_frame);

                        {
//                push input data into queue waiting to be inferred
                            std::lock_guard<std::mutex> lock(img_read_mutex);
                            img_read_queue.push(cur_frame);
                            DLOG(INFO) << "[VIDEO] " << vp.frame_id << " has been push into queue" << std::endl;

                        }

//                    }
                    vp.frame_id++;

                }

            }


            /*This part post data which has been processed*/
            OutFrame out_data;
            {

                std::lock_guard<std::mutex> lock(img_write_mutex);

                if (!img_write_queue.empty()) {
                    out_data = img_write_queue.front();
                    img_write_queue.pop();
                    DLOG(INFO) << "[VIDEO] has read " << out_data.frame_id << " output data from intp" << std::endl;
                    vp.buffer_ptr.push(out_data.input_data);
                    LOG(INFO) << "frame id:" << vp.frame_id << "; current write queue:" << img_write_queue.size();
                } else {

                    /*when read vidoe from file, end it after having read all*/
                    if (vp.frame_id == vp.video_len) {
                        LOG(INFO) << "all frame has been op" << std::endl;
                        std::chrono::time_point<std::chrono::system_clock> endTP = std::chrono::system_clock::now();

                        LOG(INFO) << " Run time: "
                                  << std::chrono::duration_cast<std::chrono::milliseconds>(endTP - startTP).count()
                                  << " ms" << std::endl;

                        vp.writer.release();
                        exit(0);
                    }

                    DLOG(INFO)<<"video not read any results from interpreter";
                    continue;
                }
            }


            /*get person's kps*/
            if (out_data.frame_id >= 0) {
                vp.out_queue.push(out_data);
                while (!vp.out_queue.empty() && last_frame_id + 1 == vp.out_queue.top().frame_id) {
                    OutFrame cur_oframe = vp.out_queue.top();
                    vp.out_queue.pop();
                    vp.ct.candidate.clear();
                    /*recover original points coordinates*/
                    vp.recoverOriginalPoints(cur_oframe.can_persons, cur_oframe.scale, cur_oframe.offset,
                                             cur_oframe.w_longer,cur_oframe.left_up);
                    vp.ct.candidate = cur_oframe.can_persons;
                    vp.ct.calCentroid();
                    vp.ct.updatePersons();
                    DLOG(INFO) << "[TRACKER] current person nums: " << vp.ct.persons.size();
                    vp.drawSingleFrame(cur_oframe);

                    if(is_falling&&!vp.ct.detect_fall){
                        char * dt= ctime(&now);
                        std::string cur_time=dt;
//                        cur_time = "#"+cur_time;
                        cur_time=cur_time.substr(0,cur_time.size()-1);
                        cur_time += "#";

                        cv::imwrite("../resources/temp.jpg",vp.video_out[cur_oframe.frame_id]);
                        std::string code=base64.base64_encode("../resources/temp.jpg");
                        cur_time += code;

                        auto pubmsg =mqtt::make_message(TOPIC_Pic,cur_time);
                        pubmsg->set_qos(1);
                        m_client.publish(pubmsg);

                        LOG(INFO)<<" video mqtt msg send: ";

                    }

                    if(vp.ct.detect_fall&&!is_falling){
                        is_falling=true;
                    }
                    vp.writer << vp.video_out[cur_oframe.frame_id];
                    vp.video_out.erase(cur_oframe.frame_id);
                    last_frame_id++;
                }

            }

//            }


        }

    }


};

