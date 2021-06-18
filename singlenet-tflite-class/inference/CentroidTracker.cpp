//
// Created by axv on 2021/5/28.
//

#include "CentroidTracker.h"

// used to compare distance for special form ( dis, (op,np) )
bool gj::cmp_dis(const std::pair<long int, cv::Point> &a, const std::pair<long int, cv::Point> &b) {
    return a.first < b.first;
}


// update candidate person's centroid and delete person with less kps
void gj::CentroidTracker::calCentroid() {
//    firstly use neck as centroid
    for (auto &p : candidate) {
//        std::cout<<p.kps.size()<<std::endl;
//        has neck as kps
        if (p.kps.find(1) != p.kps.end()) {
//            set neck as kps
            p.centroid = p.kps.find(1)->second;
            continue;
        }

//        no neck, recompute neck using R sho and L sho
        if (p.kps.find(2) != p.kps.end() && p.kps.find(5) != p.kps.end()) {
            cv::Point Rs = p.kps.find(2)->second;
            cv::Point Ls = p.kps.find(5)->second;
            p.centroid = cv::Point((Rs.x + Ls.x) / 2, (Rs.y + Ls.y) / 2);
            continue;
        }

//        no neck, lack any one of shoulder, use hip firstly
//        check R-hip and L-hip
        if (p.kps.find(8) != p.kps.end() && p.kps.find(11) != p.kps.end()) {
//            check nose
            if (p.kps.find(0) != p.kps.end()) {
                cv::Point Rh = p.kps.find(8)->second;
                cv::Point Lh = p.kps.find(11)->second;
                cv::Point No = p.kps.find(0)->second;
                p.centroid = cv::Point((Rh.x + Lh.x + No.x) / 3, (Rh.y + Lh.y + No.y) / 3);
                continue;
            }
//            check R-eye and L-eye
            if (p.kps.find(14) != p.kps.end() && p.kps.find(15) != p.kps.end()) {
                cv::Point Rh = p.kps.find(8)->second;
                cv::Point Lh = p.kps.find(11)->second;
                cv::Point Re = p.kps.find(14)->second;
                cv::Point Le = p.kps.find(15)->second;
                p.centroid = cv::Point((Rh.x + Lh.x + Re.x + Le.x) / 4, (Rh.y + Lh.y + Re.y + Le.y) / 4);
                continue;
            }
        }

//        mark not has centroid for this person
        p.centroid = cv::Point(-1, -1);

    }

//    print log to check
//    for(auto& p: candidate){
//        std::cout<<p.centroid.x<<" "<<p.centroid.y<<std::endl;
//    }
}

void gj::CentroidTracker::updatePersons() {
//    init state, no person tracking
    if (persons.empty()) {
        for (auto &p: candidate) {
//            skip person with too less kps
            if (p.kps.size() >= kp_num_thrs) {
//                add person into tracking sequence, assgin ids
                persons.insert(std::pair<long int, Person>(++cur_p_id, p));
                dis_frames.insert(std::pair<long int, int>(cur_p_id, 0));
            }
        }
        return;
    }

//    no new people and has old people
    if (candidate.empty()) {
        vector<long int> del_id;
        for (auto &p :persons) {
//            update persons disappear frames
            dis_frames.find(p.first)->second++;
//            check if old person needs to be deleted
            if (dis_frames.find(p.first)->second > dis_frames_thres) {
                del_id.push_back(p.first);
            }
        }

//        delete old people with disappearing some frames
        for (auto id:del_id) {
            persons.erase(id);
            dis_frames.erase(id);
        }
    }
//    has new people -> need to compare with old people
    else {
//        get all dis between op and np, store <d, (op,np) >
        vector<pair<long int, cv::Point>> distance;
//        unordered_set<long int> op_ids;
        set<long int> op_ids, np_ids;
        for (auto &op:persons) {
            cv::Point opc = op.second.centroid;
            for (int np = 0; np < candidate.size(); np++) {
                cv::Point npc = candidate[np].centroid;
                long int cur_dis = (opc.x - npc.x) * (opc.x - npc.x) + (opc.y - npc.y) * (opc.y - npc.y);;
                distance.emplace_back(cur_dis, cv::Point((int) op.first, np));
                op_ids.insert(op.first);
                np_ids.insert(np);
            }
        }

//        sort according to distance -> from smaller to larger
        sort(distance.begin(), distance.end(), cmp_dis);

//        record if person has been matched
        set<long int> used_op, used_np;
        for (auto &cur_dis:distance) {
//            if distance has been more than threshold, stop
            if (cur_dis.first > cd_dis_thres) {
                break;
            }
//            get ids of old person and new person
            cv::Point cur_ids = cur_dis.second;
//            person has been matched
            if (used_op.find(cur_ids.x) != used_op.end() || used_np.find(cur_ids.y) != used_np.end()) {
                continue;
            }

//            check if falls
            checkFall(persons.find(cur_ids.x)->second,candidate[cur_ids.y]);
//            update old person's data
            persons.find(cur_ids.x)->second = candidate[cur_ids.y];
//            mark this person has been used
            used_op.insert(cur_ids.x);
            used_np.insert(cur_ids.y);
        }

//        get not used old person's id
        set<long int> noused_op;
//        op_ids.
        set_difference(op_ids.begin(), op_ids.end(), used_op.begin(), used_op.end(),
                       std::inserter(noused_op, noused_op.begin()));
//        delete out frames threshold's person
        vector<long int> del_id;
        for (auto &id :noused_op) {
//            update persons disappear frames
            dis_frames.find(id)->second++;
//            check if old person needs to be deleted
            if (dis_frames.find(id)->second > dis_frames_thres) {
                del_id.push_back(id);
            }
        }

//        delete old people with disappearing some frames
        for (auto id:del_id) {
            persons.erase(id);
            dis_frames.erase(id);
        }


//        get not used new person's id
        set<long int> noused_np;
        set_difference(np_ids.begin(), np_ids.end(), used_np.begin(), used_np.end(),
                       std::inserter(noused_np, noused_np.begin()));
//        add new candidate person into trackers
        for (auto &id: noused_np) {
//            skip person with too less kps
            if (candidate[id].kps.size() >= kp_num_thrs) {
//                add person into tracking sequence, assgin ids
                persons.insert(std::pair<long int, Person>(++cur_p_id, candidate[id]));
                dis_frames.insert(std::pair<long int, int>(cur_p_id, 0));
            }
        }
    }
}

void gj::CentroidTracker::checkFall(Person &op, Person &np) {

//    about 4 fps -> so one fall only valid in one second
    if(detect_fall && ++fall_frame_cnt <5){
        return;
    }

//    reset state
    detect_fall=false;
    fall_frame_cnt=0;

    int o_w = op.right_down.x - op.left_up.x;
    int o_h = op.right_down.y - op.left_up.y;
    float n_w = np.right_down.x - np.left_up.x;
    float n_h = np.right_down.y - np.left_up.y;
    int d_x = np.centroid.x - op.centroid.x;
    int d_y = np.centroid.y - op.centroid.y;

//    bool check_fall;

    if(n_w>=1.1*n_h){
//        centroid move check
        if(sqrt(d_x*d_x+d_y*d_y) > sqrt(o_w*o_w+o_h*o_h) /2 ){
            detect_fall=true;
        }

//        specific point check
        else{
            cv::Point up_p,down_p;
            bool can_get_up=false;
            bool can_get_down=false;
//            get up points
//            1st -> neck
            if(np.kps.find(1)!=np.kps.end()){
                up_p = np.kps.find(1)->second;
                can_get_up=true;
//            2nd -> nose
            }else if(np.kps.find(0)!=np.kps.end()){
                up_p = np.kps.find(0)->second;
                can_get_up=true;
//             3rd -> midpoint of r-eye and l-eye
            }else if(np.kps.find(14)!=np.kps.end() && np.kps.find(15)!=np.kps.end()){
                up_p.x = (np.kps.find(14)->second.x + np.kps.find(15)->second.x)/2;
                up_p.y = (np.kps.find(14)->second.y + np.kps.find(15)->second.y)/2;
                can_get_up=true;
            }

            if(can_get_up){
//                get midpoint of R-hip and L-hip
                if(np.kps.find(8)!=np.kps.end() && np.kps.find(11)!=np.kps.end()){
                    down_p.x = (np.kps.find(8)->second.x + np.kps.find(11)->second.x)/2;
                    down_p.y = (np.kps.find(8)->second.y + np.kps.find(11)->second.y)/2;
                    can_get_down=true;
//                get midpoint of R-knee and L-knee
                }else if(np.kps.find(9)!=np.kps.end()&&np.kps.find(12)!=np.kps.end()){
                    down_p.x = (np.kps.find(9)->second.x + np.kps.find(12)->second.x)/2;
                    down_p.y = (np.kps.find(9)->second.y + np.kps.find(12)->second.y)/2;
                    can_get_down=true;
                }
            }

//            compare up point and down point -> if their y is one the same level -> may fall
            if(can_get_up && can_get_down){
                if(up_p.y< down_p.y+n_h/10.0){
                    detect_fall=true;
                }
            }
        }

    }
}
