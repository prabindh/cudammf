#pragma once

// Include GLEW
#include <GL/glew.h>
#include "Logger.h"

#pragma warning(disable: 4996)
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <vector>
#include <map>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

#include "decoder3_interface.h"

#include "cuda.h"
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "cuda_interop.h"

#define DECODER_MAJOR_VERSION (1)
#define DECODER_MINOR_VERSION (0)

#define MAX_SIZE_BGRA_BUFFER_Q (20)

struct TimedQ
{
    std::vector<TimedBuffer> bgraQ;
    int minTimeIndex;
};

class Decoder3 : public IDecoder3
{
public:
    Decoder3(bool bUseNvDec, bool bUseNvConverter);
    ~Decoder3();
    void GetVersion(int* maj, int* min);
    int VideoReadInit(char* fileName, int& w, int& h);
    // To be called by renderer to obtain decoded buffer
    int ReadNextFrame(TimedBuffer* pRGBBufPtr, void** pDeviceBuffer);
    // To be called by renderer after finishing usage
    int PushEmptyIntoQ(TimedBuffer pBuf, void* pDeviceBuffer);
    // CUDA convert YUV device buffer to GL device buffer
    int PushCUDAToGL(void* pDevice, int glTextureId, int bytes);
    // CUDA filter from GL texture to GL texture
    int CUDAFilterGL(int glSourceTextureId, int glDestTextureId, int w, int h, float* pFilter);
    int CUDAYUV420PToBGRA(
        void* pState,
        void* pYUV420PHost_Y,
        void* pYUV420PHost_U,
        void* pYUV420PHost_V,
        uint8_t* pBGRACpuHost,
        void** pBgraDeviceMem,
        int w, int h);
    int CUDANV12ToBGRA(
        void* pState,
        void* pYUV420PHost_Y,
        void* pYUV420PHost_UV,
        uint8_t* pBGRACpuHost,
        void** pBgraDeviceMem,
        int w, int h);
    // CUDA 
    void DeleteYUVCudaResources(void* pState);
    int CreateYUVCudaResources(void** pStatePtr, int w, int h, int format);
private:
    // Open the initial context variables that are needed
    SwsContext* img_convert_ctx;
    AVFormatContext* format_ctx;
    AVCodecContext* codec_ctx;
    int video_stream_index;
    std::ofstream output_file;
    int64_t last_time;
    int64_t first_time;
    AVPacket packet;
    AVStream* stream;
    int cntFrames;
    int readFramesCnt;
    AVCodec* codec;
    int size;
    uint8_t* picture_buffer;
    AVFrame* picture;
    AVFrame* picture_rgb;
    double timeBase;
    AVFormatContext* output_ctx;
    bool bUseNvHW_Dec;
    bool bUseNvHW_Convert;
    AVPixelFormat decoderOpFmt;

    char m_logBuff[1024];
    CPlusPlusLogging::Logger* m_pLogger;

    std::mutex m_bgraMutex;
    std::mutex m_EmptyBgraMutex;
    TimedQ m_filledBgraQ;
    std::vector<TimedBuffer> m_emptyBgraQ;
    bool m_bFirstTimeEmpty;

    std::map<int, cudaGraphicsResource_t> m_texture2CudaResourceMap;
    CudaState m_cudaState;

    void ReaderInit();
    void ReaderClose();
    int GetNextEmptyFromQ(TimedBuffer* pBufPtr);
    int GetNextFilledFromQ(TimedBuffer* pBufPtr);
    int PushFilledIntoQ(TimedBuffer pBuf);
    int UpdateMinTimeInQ();
};

#define SCALE_INDEX 25  // The index of the scale value in the filter
#define OFFSET_INDEX 26 // The index of the offset value in the filter

void CUDACopyDeviceToGL(cudaGraphicsResource_t& dstGLDeviceBuffer,
    void* srcCudaDeviceBuffer,
    unsigned int sizeBytes);
void PostprocessCUDA(cudaGraphicsResource_t& dst, cudaGraphicsResource_t& src,
    unsigned int width, unsigned int height,
    float* filter,  // Filter is assumed to be a 5x5 filter kernel
    float scale, float offset);
