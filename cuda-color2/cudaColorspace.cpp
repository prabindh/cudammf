/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "cudaColorspace.h"
#include "cudaYUV.h"




// cudaConvertColor
cudaError_t cudaConvertColorYUV420PMultiPlanar( void* inputY, void* inputU, void* inputV, imageFormat inputFormat,
					     void* output, imageFormat outputFormat,
					     size_t width, size_t height,
						 const float2& pixel_range)
{
    cudaError_t err = cudaI420MultiPlanarToRGBA(inputY, inputU, inputV, (uchar4*)output, width, height);
	return err;
}

cudaError_t cudaConvertColorNV12MultiPlanar(void* inputY, void* inputUV, imageFormat inputFormat,
	void* output, imageFormat outputFormat,
	size_t width, size_t height,
	const float2& pixel_range)
{
	cudaError_t err = cudaNV12MultiPlanarToRGBA(inputY, inputUV, (uchar4*)output, width, height);
	return err;
}


