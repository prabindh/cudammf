#include <stdio.h>
#include "windows.h"

#include "decoder3.h"

// YUV-RGB via CUDA
#include "cuda_interop.h"

/// INTERFACES

IDecoder3* GetDecoder3Obj(bool bUseNvDec, bool bUseNvConverter)
{
    Decoder3* pDec = new Decoder3(bUseNvDec, bUseNvConverter);
    return (IDecoder3*)pDec;
}
void DeleteDecoder3Obj(IDecoder3* pObj)
{
    Decoder3* pObjDec = (Decoder3*)pObj;
    if (pObjDec)
        delete pObjDec;
}



////////////////////////////////////======================
// Video related
Decoder3::Decoder3(bool bUseNvDec, bool bUseNvConverter)
    : IDecoder3(bUseNvDec, bUseNvConverter)
{
    m_pLogger = CPlusPlusLogging::Logger::getInstance();

    img_convert_ctx = NULL;
    format_ctx = NULL;
    codec_ctx = NULL;
    video_stream_index = 0;
    last_time = 0;
    first_time = 0;
    stream = NULL;
    cntFrames = 0;
    readFramesCnt = 0;
    codec = NULL;
    size = 0;
    picture_buffer = NULL;
    picture = NULL;
    picture_rgb = NULL;
    m_bFirstTimeEmpty = true;

    timeBase = 0.0;
    output_ctx = NULL;
    decoderOpFmt = AV_PIX_FMT_NONE;

    // Config
    bUseNvHW_Dec = bUseNvDec;
    bUseNvHW_Convert = bUseNvConverter;

    ReaderInit();
}
Decoder3::~Decoder3()
{
    ReaderClose();
    // Accessory CUDA objects
    for (auto cudaRes : m_texture2CudaResourceMap)
    {
        cudaGraphicsUnregisterResource(cudaRes.second);
    }
    
}
void Decoder3::GetVersion(int* maj, int* min)
{
    if (!maj || !min) return;
    *maj = DECODER_MAJOR_VERSION;
    *min = DECODER_MINOR_VERSION;
}

void Decoder3::ReaderInit()
{
    // Register everything
    av_register_all();
}

void Decoder3::ReaderClose()
{
    if (picture)
        av_frame_free(&picture);
    if (picture_rgb)
        av_frame_free(&picture_rgb);
    if (picture_buffer)
        av_free(picture_buffer);

    TimedBuffer buff;
    while (GetNextEmptyFromQ(&buff) >= 0)
    {
        av_free(buff.pBuffer);
    }
    while (GetNextFilledFromQ(&buff) >= 0)
    {
        av_free(buff.pBuffer);
    }
    m_bFirstTimeEmpty = true;

    picture = picture_rgb = NULL;
    picture_buffer = NULL;

    if (output_ctx)
    {
        avio_close(output_ctx->pb);
        avformat_free_context(output_ctx);
        output_ctx = NULL;
    }

    if (format_ctx)
    {
        av_read_pause(format_ctx);
        avformat_free_context(format_ctx);
        format_ctx = NULL;
    }

    if (img_convert_ctx)
    {
        sws_freeContext(img_convert_ctx);
        img_convert_ctx = NULL;
    }
    if (codec_ctx)
    {
        avcodec_free_context(&codec_ctx);
        codec_ctx = NULL;
    }
    if (bUseNvHW_Convert)
    {
        DeleteYUVCudaResources(&m_cudaState);
    }
    snprintf(m_logBuff, sizeof(m_logBuff), "Decoded/Read frames = %d / %d\n", cntFrames, readFramesCnt);
    m_pLogger->info(m_logBuff);
}

int Decoder3::VideoReadInit(char* fileName, int& w, int& h)
{
    format_ctx = avformat_alloc_context();
    //open
    if (avformat_open_input(&format_ctx, fileName, //"tcp://10.41.2.37:25400?listen",
        NULL, NULL) != 0) {
        return -1;
    }

    if (avformat_find_stream_info(format_ctx, NULL) < 0) {
        return -1;
    }

    //search video stream
    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
            video_stream_index = i;
    }
    av_init_packet(&packet);

    //open output file
    output_ctx = avformat_alloc_context();

    //start reading packets from stream and write them to file
    av_read_play(format_ctx);    //play TCP

    // Get the codec
    if (bUseNvHW_Dec)
    {
        codec = avcodec_find_decoder_by_name("h264_cuvid");
        // When using this - set pixel format before decode call
        // pCodecCtx->pix_fmt = AV_PIX_FMT_NV12;
        // No need to copy from output buff like for both buffers of NV12 (Y+interleaved UV)
        // Just use the pointers directly in sws_scale
    }
    else
    {
        //Generic SW decoder, can use sws scaler
        codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    }
    if (!codec) {
        return -1;
    }

    // Add this to allocate the context by codec
    codec_ctx = avcodec_alloc_context3(codec);

    avcodec_get_context_defaults3(codec_ctx, codec);
    avcodec_copy_context(codec_ctx, format_ctx->streams[video_stream_index]->codec);

    if (avcodec_open2(codec_ctx, codec, NULL) < 0)
    {
        m_pLogger->error("avcodec_open2 failed for video");
        return -1;
    }

    if (codec->capabilities & AV_CODEC_CAP_DELAY)
    {
        codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    }

    if (bUseNvHW_Convert)
    {
        if (bUseNvHW_Dec) //cuvid only nv12
        {
            decoderOpFmt = AV_PIX_FMT_NV12;
        }
        else
        {
            decoderOpFmt = AV_PIX_FMT_YUV420P; //sw only yuv420p
        }

        codec_ctx->pix_fmt = decoderOpFmt;
        // Initialise CUDA
        void* pDummy = nullptr;
        int internalPixFormat = 0; // 0 = yuv420p
        if (AV_PIX_FMT_NV12 == decoderOpFmt)
            internalPixFormat = 1;
        int retCuda = CreateYUVCudaResources(&pDummy, codec_ctx->width, codec_ctx->height, internalPixFormat);
        if (retCuda < 0)
        {
            m_pLogger->error("Could not initialise CUDA resources");
            return -1;
        }
        //FOR TESTIN ONLY - h264_cuvid can ALSO output slow to NV12 host buffers!
        if (decoderOpFmt != AV_PIX_FMT_CUDA)
        {
            img_convert_ctx = sws_getContext(codec_ctx->width, codec_ctx->height,
                codec_ctx->pix_fmt, codec_ctx->width, codec_ctx->height, AV_PIX_FMT_BGR32,
                SWS_BICUBIC, NULL, NULL, NULL);
        }
    }
    else
    {
        decoderOpFmt = AV_PIX_FMT_YUV420P;
        if (bUseNvHW_Dec)
        {
            decoderOpFmt = AV_PIX_FMT_NV12;
        }
        codec_ctx->pix_fmt = decoderOpFmt;
        img_convert_ctx = sws_getContext(codec_ctx->width, codec_ctx->height,
            codec_ctx->pix_fmt, codec_ctx->width, codec_ctx->height, AV_PIX_FMT_BGR32,
            SWS_BICUBIC, NULL, NULL, NULL);

        if (NULL == img_convert_ctx)
        {
            m_pLogger->error("Could not obtain img_convert_ctx, for given pix fmt\n");
            return -1;
        }
    }

    size = avpicture_get_size(decoderOpFmt, codec_ctx->width,
        codec_ctx->height);
    picture_buffer = (uint8_t*)(av_malloc(size));
    picture = av_frame_alloc();
    picture_rgb = av_frame_alloc();
    avpicture_fill((AVPicture *)picture, picture_buffer, decoderOpFmt,
        codec_ctx->width, codec_ctx->height);
    // Fill que of bgra converted buffers
    for (int buffId = 0; buffId < MAX_SIZE_BGRA_BUFFER_Q; buffId++)
    {
        int bgra_size2 = avpicture_get_size(AV_PIX_FMT_BGR32, codec_ctx->width,
            codec_ctx->height);
        TimedBuffer timedBuffer;
        timedBuffer.pBuffer = (uint8_t*)(av_malloc(bgra_size2));
        if (!timedBuffer.pBuffer)
        {
            snprintf(m_logBuff, sizeof(m_logBuff), "Could not allocate picture_buffer_2 % d\n", buffId);
            m_pLogger->error(m_logBuff);

            return -1;
        }
        PushEmptyIntoQ(timedBuffer, NULL);
    }
    m_filledBgraQ.minTimeIndex = 0xFFFF;

    timeBase = av_q2d(format_ctx->streams[video_stream_index]->time_base);
    //printf("VIDEO timebase: %f\n", timeBase);

    w = codec_ctx->width;
    h = codec_ctx->height;

    return 0;
}

int Decoder3::PushEmptyIntoQ(TimedBuffer rgbBuf, void* pDeviceBuff)
{
    std::unique_lock<std::mutex> lock(m_EmptyBgraMutex);
    m_emptyBgraQ.push_back(rgbBuf);

    return 0;
}


int Decoder3::GetNextEmptyFromQ(TimedBuffer* pBufPtr)
{
    std::unique_lock<std::mutex> lock(m_EmptyBgraMutex);

    if (!pBufPtr)
    {
        return -1;
    }
    if (0 == m_emptyBgraQ.size())
    {
        return -1;
    }
    *pBufPtr = m_emptyBgraQ.back();
    m_emptyBgraQ.pop_back();
    return 0;
}

int Decoder3::UpdateMinTimeInQ()
{
    // Set next lowest pts
    int64_t lowest = 0xFFFFFF;
    int lowestIndex = 0;
    for (int i = 0; i < m_filledBgraQ.bgraQ.size(); i++)
    {
        if (m_filledBgraQ.bgraQ.at(i).pts < lowest)
        {
            lowest = m_filledBgraQ.bgraQ.at(i).pts;
            lowestIndex = i;
        }
    }
    m_filledBgraQ.minTimeIndex = lowestIndex;
    return 0;
}

int Decoder3::PushFilledIntoQ(TimedBuffer pBufPtr)
{
    // push filled buffer to Q
    std::unique_lock<std::mutex> lock(m_bgraMutex);

    m_filledBgraQ.bgraQ.push_back(pBufPtr);

    UpdateMinTimeInQ();
    return 0;
}

int Decoder3::GetNextFilledFromQ(TimedBuffer* pBufPtr)
{
    // Take lowest among PTS in Q and return
    std::unique_lock<std::mutex> lock(m_bgraMutex);

    if (!pBufPtr)
    {
        return -1;
    }
    if (m_filledBgraQ.minTimeIndex < 0)
    {
        return -1;
    }
    // Wait to fill all first time
    if (m_bFirstTimeEmpty && (m_filledBgraQ.bgraQ.size() < MAX_SIZE_BGRA_BUFFER_Q))
    {
        return -1;
    }
    if (0 == m_filledBgraQ.bgraQ.size())
    {
        return -1;
    }
    m_bFirstTimeEmpty = false;

    *pBufPtr = m_filledBgraQ.bgraQ.at(m_filledBgraQ.minTimeIndex);
    m_filledBgraQ.bgraQ.erase(m_filledBgraQ.bgraQ.begin() + m_filledBgraQ.minTimeIndex);

    UpdateMinTimeInQ();

    return 0;
}
int Decoder3::ReadNextFrame(TimedBuffer* pRGBBufPtr, void** pDevicePtr)
{
    int ret = 0;
    if (!pRGBBufPtr)
    {
        return -1;
    }
    ret = av_read_frame(format_ctx, &packet);
    if (ret < 0)
    {
        m_pLogger->error("Read frame failed");
        return -2;
    }
    readFramesCnt++;
    ret = -1;
    if (packet.stream_index == video_stream_index)
    {
        ret = avcodec_send_packet(codec_ctx, &packet);
        if (ret < 0)
        {
            m_pLogger->error("avcodec_send_packet failed");
            return -1;
        }

        //packet is video
        // std::cout << "2 Is Video" << std::endl;
        if (stream == NULL) {    //create stream in file
            // std::cout << "3 create stream" << std::endl;
            stream = avformat_new_stream(output_ctx,
                format_ctx->streams[video_stream_index]->codec->codec);
            avcodec_copy_context(stream->codec,
                format_ctx->streams[video_stream_index]->codec);
            stream->sample_aspect_ratio =
                format_ctx->streams[video_stream_index]->codec->sample_aspect_ratio;
        }
        int got_picture = 0;
        packet.stream_index = stream->id;
        // std::cout << "4 decoding" << std::endl;
        //int result = avcodec_decode_video2(codec_ctx, picture, &got_picture, &packet);

        int result = avcodec_receive_frame(codec_ctx, picture);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF)
        {
            return -1;
        }
        // printf("DECODER: packet.size/result/got_picture=[%d/ %d / %d]\n", packet.size, result, got_picture);
        if (result < 0)
        {
            m_pLogger->error("video decode resulted in error");
            return -1;
        }
        //Getting all 0
        //printf("repeat_pict = %d\n", picture->repeat_pict);
        got_picture = 1;
        if (got_picture)
        {
            // timing
            last_time = (int64_t)(packet.dts * timeBase);
            if (0 == first_time)
                first_time = last_time;
            // std::cout << "Bytes decoded " << result << " check " << check << std::endl;
            // printf("DTS/PTS:  [%lld / %lld]\n", packet.dts, packet.pts);

            // Check if we have free output buff
            TimedBuffer bgraBuff;
            ret = GetNextEmptyFromQ(&bgraBuff);
            if (ret < 0)
            {
                m_pLogger->error("No output buff");
                return -1;
            }
            bgraBuff.pts = picture->pts;
            avpicture_fill((AVPicture*)picture_rgb, bgraBuff.pBuffer, AV_PIX_FMT_BGR32,
                codec_ctx->width, codec_ctx->height);

            if (bUseNvHW_Convert)
            {
                // NV12 = 2 pointers, Y, UV interl
                // pFrame->data[0](Y) and pFrame->data[1] (UV)
                if (picture->format != decoderOpFmt)
                {
                    snprintf(m_logBuff, sizeof(m_logBuff), "Mismatch in decoded and set pix_fmt[%d/%d]", 
                        picture->format, decoderOpFmt);
                    m_pLogger->error(m_logBuff);
                    ret = -1;
                }
                else
                {
                    // Convert NV12 or YUV420P to RGB
                    if (AV_PIX_FMT_YUV420P == decoderOpFmt)
                    {
                        ret = CUDAYUV420PToBGRA(&m_cudaState, picture->data[0], picture->data[1], picture->data[2],
                            bgraBuff.pBuffer, pDevicePtr,
                            codec_ctx->width, codec_ctx->height);
                    }
                    else if (AV_PIX_FMT_NV12 == decoderOpFmt)
                    {
                        ret = CUDANV12ToBGRA(&m_cudaState, picture->data[0], picture->data[1],
                            bgraBuff.pBuffer, pDevicePtr,
                            codec_ctx->width, codec_ctx->height);
                    }
                    else
                    {
                        ret = -1;
                    }
                    if (ret >= 0)
                    {
                        PushFilledIntoQ(bgraBuff);
                        // This is incorrect, as the BGRA buff is only 1, the latest, ...
                        //   and not matching with what could be returned from filledQ!
                        ret = GetNextFilledFromQ(pRGBBufPtr);
                    }
                } // if pix formats match
            }
            else
            {
                sws_scale(img_convert_ctx, picture->data, picture->linesize, 0,
                    codec_ctx->height, picture_rgb->data, picture_rgb->linesize);
                PushFilledIntoQ(bgraBuff);
                ret = GetNextFilledFromQ(pRGBBufPtr);
                //if (ret >= 0)
                //    printf("Display PTS = [%lld]\n", pRGBBufPtr->pts);
#if 0
                std::stringstream file_name;
                file_name << "test" << cnt << ".ppm";
                output_file.open(file_name.str().c_str());
                output_file << "P3 " << codec_ctx->width << " " << codec_ctx->height
                    << " 255\n";
                for (int y = 0; y < codec_ctx->height; y++) {
                    for (int x = 0; x < codec_ctx->width * 3; x++)
                        output_file
                        << (int)(picture_rgb->data[0]
                            + y * picture_rgb->linesize[0])[x] << " ";
                }
                output_file.close();
#endif // write or not to FILE
            }
            cntFrames++;
        } // got_picture
    } // if video stream
    av_packet_unref(&packet);
    av_init_packet(&packet);
    //std::cout << "Video count frames: " << cntFrames << std::endl;

    return ret;
}


int Decoder3::PushCUDAToGL(void* pDevice, int glTextureId, int w, int h)
{
    cudaGraphicsResource_t res = 0;
    cudaError_t cudaErr = cudaSuccess;
    
    // Register GL object if unregistered yet
    if (m_texture2CudaResourceMap.find(glTextureId) != m_texture2CudaResourceMap.end())
    {
        res = m_texture2CudaResourceMap[glTextureId];
    }
    else
    {
        snprintf(m_logBuff, sizeof(m_logBuff), "Creating new map for glTextureId [%d]\n", glTextureId);
        m_pLogger->info(m_logBuff);

        cudaErr = cudaGraphicsGLRegisterImage(&res, glTextureId,
            GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
        if (cudaSuccess != cudaErr)
        {
            snprintf(m_logBuff, sizeof(m_logBuff), "cudaGraphicsGLRegisterImage failed [%d]\n", cudaErr);
            m_pLogger->error(m_logBuff);

            return -1;
        }
        m_texture2CudaResourceMap[glTextureId] = res;
    }
    CUDACopyDeviceToGL(res, pDevice, w, h);

    return 0;
}
// CUDA filter from GL texture to GL texture
int Decoder3::CUDAFilterGL(int glSourceTextureId, int glDestTextureId, int w, int h, float* pFilter)
{
    cudaGraphicsResource_t resSrc = 0, resDst = 0;
    cudaError_t cudaErr = cudaSuccess;

    // Register GL object if unregistered yet
    if (m_texture2CudaResourceMap.find(glSourceTextureId) != m_texture2CudaResourceMap.end())
    {
        resSrc = m_texture2CudaResourceMap[glSourceTextureId];
    }
    else
    {
        snprintf(m_logBuff, sizeof(m_logBuff), "Creating new read map for glTextureId [%d]", glSourceTextureId);
        m_pLogger->info(m_logBuff);

        cudaErr = cudaGraphicsGLRegisterImage(&resSrc, glSourceTextureId,
            GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
        if (cudaSuccess != cudaErr)
        {
            snprintf(m_logBuff, sizeof(m_logBuff), "cudaGraphicsGLRegisterImage failed readmap[%d]\n", cudaErr);
            m_pLogger->error(m_logBuff);

            return -1;
        }
        m_texture2CudaResourceMap[glSourceTextureId] = resSrc;
    }
    // Register GL object if unregistered yet
    if (m_texture2CudaResourceMap.find(glDestTextureId) != m_texture2CudaResourceMap.end())
    {
        resDst = m_texture2CudaResourceMap[glDestTextureId];
    }
    else
    {
        snprintf(m_logBuff, sizeof(m_logBuff), "Creating new write map for glTextureId [%d]", glDestTextureId);
        m_pLogger->info(m_logBuff);

        cudaErr = cudaGraphicsGLRegisterImage(&resDst, glDestTextureId,
            GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
        if (cudaSuccess != cudaErr)
        {
            snprintf(m_logBuff, sizeof(m_logBuff), "cudaGraphicsGLRegisterImage failed for write map[%d]\n", cudaErr);
            m_pLogger->error(m_logBuff);

            return -1;
        }
        m_texture2CudaResourceMap[glDestTextureId] = resDst;
    }
    PostprocessCUDA(resDst, resSrc, w, h, pFilter, SCALE_INDEX, OFFSET_INDEX);

    return 0;
}