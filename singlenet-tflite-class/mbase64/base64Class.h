//
// Created by axv on 2021/6/11.
//

#ifndef SINGLENET_TFLITE_CLASS_BASE64CLASS_H
#define SINGLENET_TFLITE_CLASS_BASE64CLASS_H

#include <string>
#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <malloc.h>
#include <memory.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <assert.h>

typedef unsigned char uint8;
//typedef unsigned long uint32_t;

namespace gj{

    //定义编码字典
    static uint8 alphabet_map[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    //定义解码字典
    static uint8 reverse_map[] =
            {
                    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 62, 255, 255, 255, 63,
                    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 255, 255, 255, 255, 255, 255,
                    255,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 255, 255, 255, 255, 255,
                    255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 255, 255, 255, 255, 255
            };

    class base64Class {
    public:
        uint8 * encoder(const uint8* text, uint32_t text_len);
        uint8* decoder(const uint8 *code, uint32_t code_len);
        std::string base64_encode(std::string filename);
        void base64_decode(std::string filecode,  std::string filename);
    };
};


#endif //SINGLENET_TFLITE_CLASS_BASE64CLASS_H
