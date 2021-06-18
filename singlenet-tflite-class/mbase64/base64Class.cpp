//
// Created by axv on 2021/6/11.
//

#include "base64Class.h"


namespace gj{

    uint32_t encode_len_record=0;

    uint8 * base64Class::encoder(const uint8 *text, uint32_t text_len) {
        //计算解码后的数据长度
        //由以上可知  Base64就是将3个字节的数据(24位)，拆成4个6位的数据，然后前两位补零
        //将其转化为0-63的数据  然后根据编码字典进行编码
        int encode_length = text_len/3*4;
        if(text_len%3>0)
        {
            encode_length += 4;
        }

        //为编码后数据存放地址申请内存
        uint8 *encode = (uint8*)malloc(encode_length);

        memset(encode,0,encode_length);
        encode_len_record=encode_length;

//        std::cout<<"##### encoder allocate len:"<<encode_length<<std::endl;

        //编码
        uint32_t i, j;
        for (i = 0, j = 0; i+3 <= text_len; i+=3)
        {
            encode[j++] = alphabet_map[text[i]>>2];                             //取出第一个字符的前6位并找出对应的结果字符
            encode[j++] = alphabet_map[((text[i]<<4)&0x30)|(text[i+1]>>4)];     //将第一个字符的后2位与第二个字符的前4位进行组合并找到对应的结果字符
            encode[j++] = alphabet_map[((text[i+1]<<2)&0x3c)|(text[i+2]>>6)];   //将第二个字符的后4位与第三个字符的前2位组合并找出对应的结果字符
            encode[j++] = alphabet_map[text[i+2]&0x3f];                         //取出第三个字符的后6位并找出结果字符
        }

        //对于最后不够3个字节的  进行填充
        if (i < text_len)
        {
            uint32_t tail = text_len - i;
            if (tail == 1)
            {
                encode[j++] = alphabet_map[text[i]>>2];
                encode[j++] = alphabet_map[(text[i]<<4)&0x30];
                encode[j++] = '=';
                encode[j++] = '=';
            }
            else //tail==2
            {
                encode[j++] = alphabet_map[text[i]>>2];
                encode[j++] = alphabet_map[((text[i]<<4)&0x30)|(text[i+1]>>4)];
                encode[j++] = alphabet_map[(text[i+1]<<2)&0x3c];
                encode[j++] = '=';
            }
        }

//        std::cout<<"encoder final array len:"<<j<<std::endl;

        return encode;
    }

    std::string base64Class::base64_encode(std::string filename) {
        struct stat statbuff;
        if(stat(filename.c_str(), &statbuff) < 0){
            return "";
        }else{
            std::cout << "base64: file length: "<<statbuff.st_size << std::endl;
        }

        //申请一块内存  用于读取文件数据
        char *filePtr = (char *)malloc(statbuff.st_size);

        //将文件中的数据读出来
        int fd = open(filename.c_str(),O_RDONLY);
        char buffer[1024];
        int count = 0;
        if(fd > 0)
        {
            memset(buffer,0,1024);
            int len = 0;
            while ((len = read(fd,buffer,1000)) >0 )
            {
                memcpy(filePtr + count,buffer,len);
                count += len;
            }
        }
        close(fd);
        //对数据进行编码
        unsigned char* encodeData = encoder((unsigned char*)filePtr,count);

        //将编码数据放到string中 方便后面求长度
        std::string data = (char*) encodeData;
        data=data.substr(0,encode_len_record);
//        std::cout<<"##### str len: "<<data.length()<<std::endl;

        return data;
    }


    uint8 * base64Class::decoder(const uint8 *code, uint32_t code_len) {
        //由编码处可知，编码后的base64数据一定是4的倍数个字节
        assert((code_len&0x03) == 0);  //如果它的条件返回错误，则终止程序执行。4的倍数。

        //为解码后的数据地址申请内存
        uint8 *plain = (uint8*)malloc(code_len/4*3);

        //开始解码
        uint32_t i, j = 0;
        uint8 quad[4];
        for (i = 0; i < code_len; i+=4)
        {
            for (uint32_t k = 0; k < 4; k++)
            {
                quad[k] = reverse_map[code[i+k]];//分组，每组四个分别依次转换为base64表内的十进制数
            }

            assert(quad[0]<64 && quad[1]<64);

            plain[j++] = (quad[0]<<2)|(quad[1]>>4); //取出第一个字符对应base64表的十进制数的前6位与第二个字符对应base64表的十进制数的前2位进行组合

            if (quad[2] >= 64)
                break;
            else if (quad[3] >= 64)
            {
                plain[j++] = (quad[1]<<4)|(quad[2]>>2); //取出第二个字符对应base64表的十进制数的后4位与第三个字符对应base64表的十进制数的前4位进行组合
                break;
            }
            else
            {
                plain[j++] = (quad[1]<<4)|(quad[2]>>2);
                plain[j++] = (quad[2]<<6)|quad[3];//取出第三个字符对应base64表的十进制数的后2位与第4个字符进行组合
            }
        }
        return plain;
    }

    void base64Class::base64_decode(std::string filecode, std::string filename) {
        unsigned char* data=decoder((unsigned char*)filecode.c_str(),(unsigned long int)filecode.length());
        int fd_w=open(filename.c_str(),O_CREAT|O_WRONLY,S_IRUSR | S_IWUSR);
        if(fd_w > 0)
        {
            write(fd_w,(char*)data,sizeof(unsigned char)*filecode.length()/4*3);
        }
        close(fd_w);

    }

};
