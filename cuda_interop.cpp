#include <iostream>

#include "cuda-color2/cudaColorspace.h"

#include "decoder3.h"

// YUV CUDA - YUV420P,NV12 only
int Decoder3::CreateYUVCudaResources(void** pStatePtr, int w, int h, int format)
{
    cudaError_t cudaErr = cudaSuccess;
    CudaState* pState = &m_cudaState;
    int count = 0;

    cudaGetDeviceCount(&count);
    if (0 == count)
    {
        m_pLogger->error("No CUDA devices");
        return -1;
    }
    cudaSetDevice(0);

    if (!pStatePtr)
    {
        return -1;
    }
    *pStatePtr = nullptr;
    memset(pState, 0, sizeof(CudaState));

    pState->currentFormat = format;

    if (0 == w * h)
    {
        m_pLogger->error("w*h 0");
        return -1;
    }

    if (1 == format /*NV12*/)
    {
        pState->g_sizeNV12_Y = w * h;
        pState->g_sizeNV12_UV = w * h / 2;

        cudaErr = cudaMalloc(&pState->g_pDeviceNV12_Y, pState->g_sizeNV12_Y);
        if (cudaSuccess != cudaErr)
        {
            snprintf(m_logBuff, sizeof(m_logBuff), "CudaMalloc NV12Y failed [%d]", cudaErr);
            m_pLogger->error(m_logBuff);
            const char* someString;
            someString = cudaGetErrorString(cudaErr);
            if (someString) m_pLogger->error(someString);
            return -1;
        }
        cudaErr = cudaMalloc(&pState->g_pDeviceNV12_UV, pState->g_sizeNV12_UV);

        if (cudaSuccess != cudaErr)
            return -1;
    }
    else if (0 == format /* YUV420P*/)
    {
        pState->g_sizeYUV420P_Y = w * h;
        pState->g_sizeYUV420P_U = w * h / 4;
        pState->g_sizeYUV420P_V = w * h / 4;
        cudaErr = cudaMalloc(&pState->g_pDeviceYUV420P_Y, pState->g_sizeYUV420P_Y);
        if (cudaSuccess != cudaErr)
        {
            snprintf(m_logBuff, sizeof(m_logBuff), "CudaMalloc 420Y failed [%d]", cudaErr);
            m_pLogger->error(m_logBuff);
            const char* someString;
            someString = cudaGetErrorString(cudaErr);
            if (someString) m_pLogger->error(someString);
            return -1;
        }
        cudaErr = cudaMalloc(&pState->g_pDeviceYUV420P_U, pState->g_sizeYUV420P_U);
        if (cudaSuccess != cudaErr)
            return -1;
        cudaErr = cudaMalloc(&pState->g_pDeviceYUV420P_V, pState->g_sizeYUV420P_V);
        if (cudaSuccess != cudaErr)
            return -1;

    }
    else
    {
        return -1;
    }
    pState->g_sizeBGRA = w * h * 4;

    // Allocate GPU friendly page-locked host mem
    cudaErr = cudaMallocHost(&pState->g_pHostCudaAlignedBGRA, pState->g_sizeBGRA);
    if (cudaSuccess != cudaErr)
        return -1;

    // Allocate device mem
    cudaErr = cudaMalloc(&pState->g_pDeviceBGRA, pState->g_sizeBGRA);
    if (cudaSuccess != cudaErr)
        return -1;

    *pStatePtr = &m_cudaState;
    return 0;
}
void Decoder3::DeleteYUVCudaResources(void *pCudaState)
{
    CudaState* pState = (CudaState*)pCudaState;
    // Free
    cudaFree(pState->g_pDeviceBGRA); pState->g_pDeviceBGRA = NULL;
    cudaFreeHost(pState->g_pHostCudaAlignedBGRA); pState->g_pHostCudaAlignedBGRA = NULL;
    // video buffers
    cudaFree(pState->g_pDeviceNV12_Y); pState->g_pDeviceNV12_Y = NULL;
    cudaFree(pState->g_pDeviceNV12_UV); pState->g_pDeviceNV12_UV = NULL;
    cudaFree(pState->g_pDeviceYUV420P_Y); pState->g_pDeviceYUV420P_Y = NULL;
    cudaFree(pState->g_pDeviceYUV420P_U); pState->g_pDeviceYUV420P_U = NULL;
    cudaFree(pState->g_pDeviceYUV420P_V); pState->g_pDeviceYUV420P_V = NULL;
}


//  on Host, CUDA processing to RGB GL on device directly
int Decoder3::CUDAYUV420PToBGRA(void* pCudaState, void* pYUV420PHost_Y, void* pYUV420PHost_U, void* pYUV420PHost_V,
    uint8_t* pBGRACpuHost, void** pBGRADevice,int w, int h)
{
    cudaError_t cudaErr = cudaSuccess;

    CudaState* pState = (CudaState*)pCudaState;
    // Some sanity
    if (pState->g_sizeBGRA != w * h * 4)
    {
        snprintf(m_logBuff, sizeof(m_logBuff), "Mismatch in global state wh4 = [%d], g_sizeBGRA = [%d]", w * h * 4, pState->g_sizeBGRA);
        m_pLogger->error(m_logBuff);
        return -1;
    }
    if (!pBGRACpuHost)
        return -1;

    // Copy input data to device mem
    cudaErr = cudaMemcpy(pState->g_pDeviceYUV420P_Y, pYUV420PHost_Y, pState->g_sizeYUV420P_Y, cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess)
    {
        m_pLogger->error("Error in memcpy - host to device 420p");
        return -1;
    }
    cudaErr = cudaMemcpy(pState->g_pDeviceYUV420P_U, pYUV420PHost_U, pState->g_sizeYUV420P_U, cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess)
    {
        m_pLogger->error("Error in memcpy U- host to device 420p");
        return -1;
    }
    cudaErr = cudaMemcpy(pState->g_pDeviceYUV420P_V, pYUV420PHost_V, pState->g_sizeYUV420P_V, cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess)
    {
        m_pLogger->error("Error in memcpy V- host to device 420p");
        return -1;
    }

    // Run RGB conversion kernel
    // Result in device memory
    cudaErr = cudaConvertColorYUV420PMultiPlanar(pState->g_pDeviceYUV420P_Y,
        pState->g_pDeviceYUV420P_U, pState->g_pDeviceYUV420P_V,
        IMAGE_I420,
        pState->g_pDeviceBGRA,
        IMAGE_RGBA8,
        w, h);

    if (cudaErr != cudaSuccess)
    {
        snprintf(m_logBuff, sizeof(m_logBuff), "CUDA Error in kernel - [%d]", cudaErr);
        m_pLogger->error(m_logBuff);

        return -1;
    }
    if (pBGRADevice)
    {
        *pBGRADevice = pState->g_pDeviceBGRA;
    }
    else
    {
        // Copy RGB data to CPU
        cudaErr = cudaMemcpy(pBGRACpuHost, pState->g_pDeviceBGRA, pState->g_sizeBGRA, cudaMemcpyDeviceToHost);
        if (cudaErr != cudaSuccess)
        {
            snprintf(m_logBuff, sizeof(m_logBuff), "Error in memcpy - device to host BGRA [%d]\n", cudaErr);
            m_pLogger->error(m_logBuff);

            return -1;
        }
    }
    //memset(pBGRACpuHost, 0xFF, pState->g_sizeBGRA);

    return 0;
}



//  on Host, CUDA processing to RGB GL on device directly
int Decoder3::CUDANV12ToBGRA(void* pCudaState, void* pYUV420PHost_Y, void* pYUV420PHost_UV,
    uint8_t* pBGRACpuHost, void** pBGRADevice, int w, int h)
{
    cudaError_t cudaErr = cudaSuccess;

    CudaState* pState = (CudaState*)pCudaState;
    // Some sanity
    if (pState->g_sizeBGRA != w * h * 4)
    {
        snprintf(m_logBuff, sizeof(m_logBuff), "Mismatch in global state wh4 = [%d], g_sizeBGRA = [%d]", w * h * 4, pState->g_sizeBGRA);
        m_pLogger->error(m_logBuff);
        return -1;
    }
    if (!pBGRACpuHost)
        return -1;

    // Copy input data to device mem
    cudaErr = cudaMemcpy(pState->g_pDeviceNV12_Y, pYUV420PHost_Y, pState->g_sizeNV12_Y, cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess)
    {
        m_pLogger->error("Error in memcpy - host to device NV12 Y");
        return -1;
    }
    cudaErr = cudaMemcpy(pState->g_pDeviceNV12_UV, pYUV420PHost_UV, pState->g_sizeNV12_UV, cudaMemcpyHostToDevice);
    if (cudaErr != cudaSuccess)
    {
        m_pLogger->error("Error in memcpy U- host to device nv12 uv");
        return -1;
    }

    // Run RGB conversion kernel
    // Result in device memory
    cudaErr = cudaConvertColorNV12MultiPlanar(pState->g_pDeviceYUV420P_Y,
        pState->g_pDeviceNV12_UV,
        IMAGE_NV12,
        pState->g_pDeviceBGRA,
        IMAGE_RGBA8,
        w, h);

    if (cudaErr != cudaSuccess)
    {
        snprintf(m_logBuff, sizeof(m_logBuff), "CUDA Error in NV12 kernel - [%d]", cudaErr);
        m_pLogger->error(m_logBuff);

        return -1;
    }
    if (pBGRADevice)
    {
        *pBGRADevice = pState->g_pDeviceBGRA;
    }
    else
    {
        // Copy RGB data to CPU
        cudaErr = cudaMemcpy(pBGRACpuHost, pState->g_pDeviceBGRA, pState->g_sizeBGRA, cudaMemcpyDeviceToHost);
        if (cudaErr != cudaSuccess)
        {
            snprintf(m_logBuff, sizeof(m_logBuff), "Error in memcpy - device to host BGRA [%d]\n", cudaErr);
            m_pLogger->error(m_logBuff);

            return -1;
        }
    }
    //memset(pBGRACpuHost, 0xFF, pState->g_sizeBGRA);

    return 0;
}
