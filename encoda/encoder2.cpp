#include "encoder.h"

ENCODA_API IEncoder2* GetEncoder2Obj(const bool hwAccel, const bool bNv,
    int w, int h, int fps, const char* outFile, int format)
{
    Encoder2* pEnc = new Encoder2(hwAccel, bNv, w, h, fps, outFile, format);
    return (IEncoder2*)pEnc;
}
ENCODA_API void DeleteEncoder2Obj(IEncoder2* pObj)
{
    Encoder2* pEnc = (Encoder2*)pObj;
    delete pEnc;
}

/// <summary>
/// Encoder Implementation
/// </summary>
/// <param name="bHwAccel"></param>
/// <param name="bNv"></param>
/// <param name="w"></param>
/// <param name="h"></param>
/// <param name="fps"></param>
/// <param name="outFile"></param>
/// <param name="fmt"></param>

Encoder2::Encoder2(const bool bHwAccel, const bool bNv, 
                int w, int h, int fps, const char* outFile, int fmt)
    :IEncoder2(bHwAccel, bNv, w, h, fps, outFile, fmt)
{
    int ret = 0;
    
    m_bInited = false;
    if (fmt == 0) // 420P supported
    {
        m_pixFormat = AV_PIX_FMT_YUV420P;
    }
    else if (fmt == 1) // nv12 
    {
        m_pixFormat = AV_PIX_FMT_NV12;
    }
    else
    {
        return;
    }
    

    m_frameId = 0;
    m_width = w;
    m_height = h;
    m_fps = fps;
    m_muxer = 0;
    m_bUseFilterGraph = true;
    m_numFiltersInGraph = 0;

    ret = setup(outFile, bHwAccel, bNv);
    if (ret >= 0)
    {
        m_bInited = true;
    }
}

Encoder2::~Encoder2()
{

}

bool Encoder2::isInited()
{
    return m_bInited;
}

int Encoder2::createFilterGraphPureSW(AVPixelFormat pixFormat)
{
    int ret = 0;

    const char* filterNames[MAX_PURESW_FILTERS] = { "buffer","format",
        "scale", "format", "buffersink" };
    const char* filterSelfNames[MAX_PURESW_FILTERS] = { "in","",
            "", "", "out" };
    const char* filterArgs[MAX_PURESW_FILTERS] = { "", "pix_fmts=yuv420p", "x:y", "pix_fmts=yuv420p", "" };
    void* filterParams[MAX_PURESW_FILTERS] = { 0 };

    if (AV_PIX_FMT_NV12 == pixFormat) // nv12
    {
        filterArgs[1] =  filterArgs[3] = "pix_fmts=nv12";
    }

    AVFilterGraph *graph = avfilter_graph_alloc();

    // input args
    char args[512];
    snprintf(args, sizeof(args),
        "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
        m_width, m_height, m_pixFormat,
        1, m_fps, 1, 1);
    filterArgs[0] = args;

    // scale args
    char scaleArgs[512];
    snprintf(scaleArgs, sizeof(scaleArgs), "%d:%d", m_width, m_height);
    filterArgs[2] = scaleArgs;

    // output params
    AVPixelFormat formatList[2] = { m_pixFormat, AV_PIX_FMT_NONE };
    AVBufferSinkParams buffersink_params;
    buffersink_params.pixel_fmts = formatList;
    filterParams[MAX_PURESW_FILTERS - 1] = &buffersink_params;

    // Create filters
    for (int i = 0; i < MAX_PURESW_FILTERS; i++)
    {
        ret = avfilter_graph_create_filter(&m_filterContexts[i],
            avfilter_get_by_name(filterNames[i]), filterSelfNames[i], filterArgs[i], filterParams[i], graph);
        if (ret < 0)
        {
            return -1;
        }
    }
    // link
    for (int i = 0; i < MAX_PURESW_FILTERS - 1; i++)
    {
        ret = avfilter_link(m_filterContexts[i], 0, m_filterContexts[i + 1], 0);
    }
    // Finalise
    ret = avfilter_graph_config(graph, 0);

    m_numFiltersInGraph = MAX_PURESW_FILTERS;

    return ret;
}
int Encoder2::createFilterGraphNv(AVPixelFormat pixFormat)
{
    int ret = 0;

    const char* filterNames[MAX_FILTERS] = {"buffer","format", "hwupload_cuda", 
        "scale_cuda", "hwdownload", "format", "buffersink"};
    const char* filterSelfNames[MAX_FILTERS] = { "in","", "",
            "", "", "", "out" };
    const char* filterArgs[MAX_FILTERS] = {"", "pix_fmts=yuv420p", "", "x:y", "", "pix_fmts=yuv420p", "" };
    void* filterParams[MAX_FILTERS] = {0};

    AVFilterGraph *graph = avfilter_graph_alloc();

    if (AV_PIX_FMT_NV12 == pixFormat) // nv12
    {
        filterArgs[1] = filterArgs[5] = "pix_fmts=nv12";
    }

    // input args
    char args[512];
    snprintf(args, sizeof(args),
        "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
        m_width, m_height, m_pixFormat,
        1, m_fps, 1, 1);
    filterArgs[0] = args;

    // scale args
    char scaleArgs[512];
    snprintf(scaleArgs, sizeof(scaleArgs), "%d:%d", m_width, m_height);
    filterArgs[3] = scaleArgs;

    // output params
    AVPixelFormat formatList[2] = { m_pixFormat, AV_PIX_FMT_NONE };
    AVBufferSinkParams buffersink_params;
    buffersink_params.pixel_fmts = formatList;
    filterParams[MAX_FILTERS-1] = &buffersink_params;

    // Create filters
    for (int i = 0; i < MAX_FILTERS; i++)
    {
        ret = avfilter_graph_create_filter(&m_filterContexts[i],
            avfilter_get_by_name(filterNames[i]), filterSelfNames[i], filterArgs[i], filterParams[i], graph);
        if (ret < 0)
        {
            return -1;
        }
    }
    // link
    for (int i = 0; i < MAX_FILTERS-1; i++)
    {
        ret = avfilter_link(m_filterContexts[i], 0, m_filterContexts[i+1], 0);
    }
    // Finalise
    ret = avfilter_graph_config(graph, 0);

    m_numFiltersInGraph = MAX_FILTERS;

    return ret;
}

int Encoder2::addFrame(uint8_t* pData)
{
    int ret = 0;
    AVFrame* input = nullptr;

    input = av_frame_alloc();
    ret = avpicture_fill((AVPicture *)input, pData, m_pixFormat,
                            m_width, m_height);
    input->format = m_pixFormat;
    input->width = m_width;
    input->height = m_height;

    if (m_bUseFilterGraph)
    {
        AVFrame* out = nullptr;
        out = av_frame_alloc();

        ret = av_buffersrc_write_frame(m_filterContexts[0], input);
        if (ret < 0)
        {
            av_frame_free(&out);
            av_frame_free(&input);
            return -1;
        }
        if (0 == m_numFiltersInGraph)
        {
            av_frame_free(&out);
            av_frame_free(&input);
            return -1;
        }
        ret = av_buffersink_get_frame(m_filterContexts[m_numFiltersInGraph - 1], out);
        if (ret < 0)
        {
            av_frame_free(&out);
            av_frame_free(&input);
            return -1;
        }

        out->pts = m_frameId++;

        ret = encodeFrame(out);

        if (out)
            av_frame_free(&out);
    }
    else
    {
        input->pts = m_frameId++;
        ret = encodeFrame(input);
    }
    av_frame_free(&input);

    return ret;
}

int Encoder2::encodeFrame(AVFrame* frame)
{
    int ret = avcodec_send_frame(m_encoder, frame);
    
    if (ret < 0)
    {
        return -1;
    }

    AVPacket packet;
    av_init_packet(&packet);

    while (ret >= 0) 
    {
        ret = avcodec_receive_packet(m_encoder, &packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            return 0;  // nothing to write
        }
        if (ret == 0)
        {
            av_packet_rescale_ts(&packet, m_encoder->time_base, m_avStream->time_base);
            packet.stream_index = m_avStream->index;
            av_interleaved_write_frame(m_muxer, &packet);
            av_packet_unref(&packet);
        }
    }
    return 0;
}

int Encoder2::flush()
{
    int ret = 0;
    
    ret = encodeFrame(nullptr);
    
    ret = av_write_trailer(m_muxer);
    
    return 0;
}


int Encoder2::setup(const char* outFile, const bool bHwAccel, const bool bNv)
{
    AVOutputFormat * outFmt = av_guess_format("mp4", NULL, NULL);
    int ret = avformat_alloc_output_context2(&m_muxer, outFmt, nullptr, nullptr);
    
    if (ret < 0 || !m_muxer)
    {
        return -1;
    }

    ret = setupEncoder(bHwAccel, bNv);
    if (ret < 0)
    {
        return -1;
    }
    m_avStream = avformat_new_stream(m_muxer, nullptr);
    assert(m_avStream != nullptr);
    if (!m_avStream)
    {
        return -1;
    }

    m_avStream->id = m_muxer->nb_streams - 1;
    m_avStream->time_base = m_encoder->time_base;

    // Some formats want stream headers to be separate.
    if (m_muxer->oformat->flags & AVFMT_GLOBALHEADER)
        m_encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    ret = avcodec_parameters_from_context(m_avStream->codecpar, m_encoder);
    if (ret < 0)
    {
        return -1;
    }

    ret = avio_open(&m_muxer->pb, outFile, AVIO_FLAG_WRITE);
    if (ret < 0)
    {
        return -1;
    }

    ret = avformat_write_header(m_muxer, nullptr);
    
    return ret;
}

int Encoder2::setupEncoder(const bool bHwAccel, const bool bNv)
{
    int ret = 0;
    const char* encoderName = nullptr;
    AVCodec* videoCodec = nullptr;
    AVRational timeBase, frameRate;

    if (bNv)
    {
        encoderName = "h264_nvenc";
    }
    else
    {
        encoderName = "libx264";
    }

    videoCodec = avcodec_find_encoder_by_name(encoderName);
    if (!videoCodec)
    {
        return -1;
    }
    m_encoder = avcodec_alloc_context3(videoCodec);

    m_encoder->bit_rate = m_width * m_height * m_fps * 2;
    m_encoder->width = m_width;
    m_encoder->height = m_height;

    m_encoder->gop_size = m_fps;
    m_encoder->max_b_frames = 1;

    timeBase = { 1, m_fps };
    frameRate = { m_fps, 1 };
    m_encoder->time_base = timeBase;
    m_encoder->framerate = frameRate;

    m_encoder->gop_size = m_fps;  // have at least 1 I-frame per second
    m_encoder->max_b_frames = 1;
    m_encoder->pix_fmt = m_pixFormat;

    // Create all filters in graph
    if (m_bUseFilterGraph && bNv)
    {
        createFilterGraphNv(m_pixFormat);
    }
    else if (m_bUseFilterGraph)
    {
        createFilterGraphPureSW(m_pixFormat);
    }

    ret = avcodec_open2(m_encoder, videoCodec, nullptr);
    if (ret < 0)
    {
        return -1;
    }
    m_muxer->video_codec_id = videoCodec->id;
    m_muxer->video_codec = videoCodec;

    return 0;
}