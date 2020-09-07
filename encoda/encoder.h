#pragma once

#include "encoder_interface.h"
#include <cassert>

#pragma warning(disable: 4996)

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>

#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

#include <thread>
#include <vector>
#include <mutex>
#include "../Logger.h"

#define ENCODER_MAJOR_VERSION (1)
#define ENCODER_MINOR_VERSION (0)


struct DataBuff
{
    void* pBuff;
    int sizeBytes;
    int pixFormat;
};

#define MAX_FILTERS 7
#define MAX_PURESW_FILTERS (5)
class Encoder2 : public IEncoder2
{
public:
    Encoder2(const bool hwAccel, const bool bNv, int w, int h, int fps, const char* outFile, int format);
    ~Encoder2();
    void GetVersion(int* maj, int* min);
    bool IsInited();
    int AddFrame(uint8_t* pBuffer);
    int AddFrameToQ(uint8_t* pBuffer, int sizeBytes);
    int Flush();
private:
    AVFilterContext* m_filterContexts[MAX_FILTERS] = { 0 };
    int m_frameId;

    AVCodecContext* m_encoder;
    AVFormatContext* m_muxer;
    AVStream* m_avStream;
    AVFrame* m_frameHeader;
    AVPixelFormat m_pixFormat;
    AVFilterGraph* m_graph;

    char m_logBuff[1024];
    CPlusPlusLogging::Logger* pLogger;

    int m_width ;
    int m_height;
    int m_fps ;
    bool m_bInited;
    bool m_bUseFilterGraph;
    int m_numFiltersInGraph;
    int m_rawBufferSizeBytes;

    int createFilterGraphNv(AVPixelFormat pixFormat);
    int createFilterGraphPureSW(AVPixelFormat pixFormat);
    int encodeFrame(AVFrame* frame);
    int setup(const char* outFile, const bool bHwAccel, const bool bNv);
    int setupEncoder(const bool bHwAccel, const bool bNv);
    // Thread
    std::thread m_encodeThread;
#define MAX_PENDING_ENCODE_ITEMS (10)
    std::vector<DataBuff> m_encodeInputQ;
    std::mutex m_inputQMutex;
    std::condition_variable m_encodeThreadCv;
    bool m_bEncoderStopped;
    void EncodeThreadFunc(void* param);
};