void PostprocessCUDA( cudaGraphicsResource_t& dst, cudaGraphicsResource_t& src, 
                     unsigned int width, unsigned int height, 
                     float* filter,             // Filter is assumed to be a 5x5 filter kernel
                     float scale, float offset ); 

void CUDACopyDeviceToGL(cudaGraphicsResource_t& dstGLDeviceBuffer,
    void* srcCudaDeviceBuffer,
    unsigned int sizeBytes);