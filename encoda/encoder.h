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

#define MAX_FILTERS 7
#define MAX_PURESW_FILTERS (5)
class Encoder2 : public IEncoder2
{
public:
    Encoder2(const bool hwAccel, const bool bNv, int w, int h, int fps, const char* outFile, int format);
    ~Encoder2();
    bool isInited();
    int addFrame(uint8_t* pBuffer);
    int flush();
private:
    AVFilterContext* m_filterContexts[MAX_FILTERS] = { 0 };
    int m_frameId;

    AVCodecContext* m_encoder;
    AVFormatContext* m_muxer;
    AVStream* m_avStream;
    AVFrame* m_frameHeader;
    AVPixelFormat m_pixFormat;

    int m_width ;
    int m_height;
    int m_fps ;
    bool m_bInited;
    bool m_bUseFilterGraph;
    int m_numFiltersInGraph;

    int createFilterGraphNv(AVPixelFormat pixFormat);
    int createFilterGraphPureSW(AVPixelFormat pixFormat);
    int encodeFrame(AVFrame* frame);
    int setup(const char* outFile, const bool bHwAccel, const bool bNv);
    int setupEncoder(const bool bHwAccel, const bool bNv);
};