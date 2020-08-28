#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#define DECODA_API __declspec(dllexport)
#else
#define DECODA_API __attribute__((visibility("default")))
#endif

struct TimedBuffer
{
    uint8_t* pBuffer;
    int64_t pts;
};

class IDecoder3
{
public:
    IDecoder3(bool bUseNvDec, bool bUseNvConverter) {};
    virtual ~IDecoder3() {};
    virtual int VideoReadInit(char* fileName, int& w, int& h) = 0;
    // To be called by renderer to obtain decoded buffer
    virtual int ReadNextFrame(TimedBuffer* pRGBBufPtr, void** pDeviceBuffer) = 0;
    // To be called by renderer after finishing usage
    virtual int PushEmptyIntoQ(TimedBuffer pBuf, void* pDeviceBuffer) = 0;
    // CUDA init (for direct cuda-usage without decoder only)
    virtual int CreateYUVCudaResources(void** pStatePtr, int w, int h, int format) = 0;
    // CUDA convert YUV420P to BGRA device - requires CreateYUVCudaResources call and cleanup
    virtual int CUDAYUV420PToBGRA(
        void* pState,
        void* pYUV420PHost_Y,
        void* pYUV420PHost_U,
        void* pYUV420PHost_V,
        uint8_t* pBGRACpuHost,
        void** pBgraDeviceMem,
        int w, int h) = 0;
    virtual int CUDANV12ToBGRA(
        void* pState,
        void* pYUV420PHost_Y,
        void* pYUV420PHost_UV,
        uint8_t* pBGRACpuHost,
        void** pBgraDeviceMem,
        int w, int h) = 0;
    // CUDA push BGRA device buffer to GL device buffer - does NOT need init/cleanup
    virtual int PushCUDAToGL(void* pDeviceMem, int glTextureId, int bytes) = 0;
    // CUDA filter from GL texture to GL texture - does NOT need init
    virtual int CUDAFilterGL(int glSourceTextureId, int glDestTextureId, int w, int h, float* pFilter) = 0;
    // CLEANUP (for direct cuda - usage without decoder only)
    virtual void DeleteYUVCudaResources(void* pState) = 0;
};

// Factory
extern "C" DECODA_API IDecoder3* GetDecoder3Obj(bool bUseNvDec, bool bUseNvConverter);
extern "C" DECODA_API void DeleteDecoder3Obj(IDecoder3* pObj);

typedef IDecoder3* (*PFN_GETDECODER3OBJ)(bool bUseNvDec, bool bUseNvConverter);
typedef void (*PFN_DELETEDECODER3OBJ)(IDecoder3* pObj);
