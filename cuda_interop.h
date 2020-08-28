#pragma once

struct CudaState
{
    int currentFormat = 1;
    uint8_t* g_pDeviceBGRA = NULL;
    unsigned int g_sizeBGRA = 0;
    uint8_t* g_pHostCudaAlignedBGRA = NULL;
    uint8_t* g_pDeviceNV12_Y = NULL;
    uint8_t* g_pDeviceNV12_UV = NULL;
    unsigned int g_sizeNV12 = 0;
    unsigned int g_sizeNV12_Y = 0;
    unsigned int g_sizeNV12_UV = 0;

    uint8_t* g_pDeviceYUV420P_Y = NULL;
    uint8_t* g_pDeviceYUV420P_U = NULL;
    uint8_t* g_pDeviceYUV420P_V = NULL;
    unsigned int g_sizeYUV420P_Y = 0;
    unsigned int g_sizeYUV420P_U = 0;
    unsigned int g_sizeYUV420P_V = 0;
};


// GL-CUDA
