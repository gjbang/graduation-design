#include <iostream>
#include "kputil.h"
#include <thread>

using namespace std;
using namespace gj;
using namespace google;

int main() {

    // init glog setting
    FLAGS_log_dir = "../logs/";
    FLAGS_colorlogtostderr =true;
    FLAGS_alsologtostderr = true;
    google::InitGoogleLogging("glogs");

    LOG(INFO) << "start to run task pool" << endl;

    constexpr unsigned num_threads = 6;
    int cpu_id = 0;
    std::vector<std::thread> task_pool;

    for (auto iter = mapThreadMainManager.begin(); iter != mapThreadMainManager.end(); ++iter, ++cpu_id) {
        std::cout << iter->first << std::endl;

        if(iter->first=="mqtt"){
            cpu_id--;
        }

        ThreadMain func = iter->second;
        std::thread thd(iter->second);

        if (BIND_CPU) {
            LOG(INFO) << " use bind cpu mode" << endl;

            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpu_id, &cpuset);
            int rc = pthread_setaffinity_np(thd.native_handle(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0) {
                LOG(ERROR) << " Error calling pthread_setaffinity_np: " << rc << endl;
            } else {
                LOG(INFO) << " CPU-" << cpu_id << " has been bind with " << iter->first << endl;
            }

        } else {
            LOG(INFO) << " not use bind-cpu mode" << endl;
        }

        task_pool.push_back(std::move(thd));
    }


    for (auto iter = task_pool.begin(); iter != task_pool.end(); ++iter) {
        iter->join();
    }

    std::this_thread::sleep_until(
            std::chrono::time_point<std::chrono::system_clock>::max());

    google::ShutdownGoogleLogging();
    LOG(INFO)<<"Program exit normally";

    return 0;
}
