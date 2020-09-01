#pragma once

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#define ENCODA_API __declspec(dllexport)
#else
#define ENCODA_API __attribute__((visibility("default")))
#endif

class IEncoder2
{
public:
    IEncoder2(const bool hwAccel, const bool bNv, int w, int h, int fps, const char* outFile, int format) {};
    virtual ~IEncoder2() {};
    virtual void GetVersion(int* maj, int* min) = 0;
    virtual bool isInited() = 0;
    virtual int addFrame(uint8_t* pBuffer) = 0;
    virtual int addFrameToQ(uint8_t* pBuffer, int sizeBytes) = 0;
};

// Factory
extern "C" ENCODA_API IEncoder2 * GetEncoder2Obj(const bool hwAccel, const bool bNv,
                int w, int h, int fps, const char* outFile, int format);
extern "C" ENCODA_API void DeleteEncoder2Obj(IEncoder2 * pObj);

typedef IEncoder2* (*PFN_GETENCODER2OBJ)(const bool hwAccel, const bool bNv,
    int w, int h, int fps, const char* outFile, int format);
typedef void (*PFN_DELETEENCODER2OBJ)(IEncoder2* pObj);

