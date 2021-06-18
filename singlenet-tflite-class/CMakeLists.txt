cmake_minimum_required(VERSION 3.16)
project(singlenet_tflite_class)

set(CMAKE_CXX_STANDARD 14)

INCLUDE_DIRECTORIES(util)
INCLUDE_DIRECTORIES(inference)
INCLUDE_DIRECTORIES(video)
INCLUDE_DIRECTORIES(mqtt)
INCLUDE_DIRECTORIES(mbase64)
INCLUDE_DIRECTORIES(include/)
INCLUDE_DIRECTORIES(/usr/local/include/opencv4)


#LINK_LIBRARIES(${CMAKE_SOURCE_DIR}/lib)
LINK_LIBRARIES(tensorflow-lite pthread dl flatbuffers glog)
LINK_LIBRARIES(opencv_core opencv_calib3d opencv_dnn opencv_gapi opencv_imgproc opencv_highgui opencv_videoio opencv_imgcodecs opencv_features2d opencv_flann opencv_imgproc opencv_ml opencv_objdetect opencv_photo opencv_bgsegm opencv_video)
LINK_LIBRARIES(paho-mqttpp3 paho-mqtt3c paho-mqtt3a)
LINK_LIBRARIES(${OpenCV_LIBS})

add_executable(singlenet_tflite_class main.cpp util/kputil.cpp util/kputil.h inference/KPDetect.cpp inference/KPDetect.h video/VideoProcess.cpp video/VideoProcess.h inference/FallCheck.cpp inference/FallCheck.h inference/CentroidTracker.cpp inference/CentroidTracker.h mqtt/smqtt.cpp mqtt/smqtt.h mbase64/base64Class.cpp mbase64/base64Class.h)