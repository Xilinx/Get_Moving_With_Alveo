/**********
Copyright (c) 2019, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/


#include "ap_int.h"
#include "common/xf_common.h"
#include "common/xf_utility.h"
#include "hls_stream.h"
#include "imgproc/xf_gaussian_filter.hpp"
#include "imgproc/xf_resize.hpp"


#define AXI_WIDTH 512
#define TYPE XF_8UC3
#define NPC XF_NPPC8

#define PRAGMA_SUB(x) _Pragma(#x)
#define DYN_PRAGMA(x) PRAGMA_SUB(x)

#define MAX_IN_WIDTH 3840
#define MAX_IN_HEIGHT 2160
#define MAX_OUT_WIDTH 3840
#define MAX_OUT_HEIGHT 2160

#define STREAM_DEPTH 8

#define MAX_DOWN_SCALE 7
#define FILTER_WIDTH 7

extern "C"
{

    void mat_split(xf::Mat<XF_8UC3, MAX_IN_HEIGHT, MAX_IN_WIDTH, NPC> &_src,
                   xf::Mat<XF_8UC1, MAX_IN_HEIGHT, MAX_IN_WIDTH, NPC> &_r,
                   xf::Mat<XF_8UC1, MAX_IN_HEIGHT, MAX_IN_WIDTH, NPC> &_g,
                   xf::Mat<XF_8UC1, MAX_IN_HEIGHT, MAX_IN_WIDTH, NPC> &_b)
    {
        ap_uint<24 * NPC> din;
        ap_uint<8 * NPC> r, g, b;
#pragma HLS INLINE
    mat_split_x:
        for (int x = 0; x < _src.rows; x++) {
            DYN_PRAGMA(HLS LOOP TRIPCOUNT max = MAX_IN_HEIGHT)
        mat_split_y:
            for (int y = 0; y < (_src.cols >> XF_BITSHIFT(NPC)); y++) {
                DYN_PRAGMA(HLS LOOP TRIPCOUNT max = MAX_IN_WIDTH)
#pragma HLS PIPELINE
                din = _src.data[x * (_src.cols >> XF_BITSHIFT(NPC)) + y];
            mat_split_i:
                for (int i = 0; i < (1 << XF_BITSHIFT(NPC)); i++) {
#pragma HLS UNROLL
                    r(i * 8 + 7, i * 8) = din.range(23 + i * 24, 16 + i * 24);
                    g(i * 8 + 7, i * 8) = din.range(15 + i * 24, 8 + i * 24);
                    b(i * 8 + 7, i * 8) = din.range(7 + i * 24, 0 + i * 24);
                }
                _r.data[x * (_src.cols >> XF_BITSHIFT(NPC)) + y] = r;
                _g.data[x * (_src.cols >> XF_BITSHIFT(NPC)) + y] = g;
                _b.data[x * (_src.cols >> XF_BITSHIFT(NPC)) + y] = b;
            }
        }
    }

    void mat_combine(xf::Mat<XF_8UC1, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> &_r,
                     xf::Mat<XF_8UC1, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> &_g,
                     xf::Mat<XF_8UC1, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> &_b,
                     xf::Mat<XF_8UC3, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> &_dest)
    {
        ap_uint<24 * NPC> dout;
        ap_uint<8 * NPC> r, g, b;
#pragma HLS INLINE
        for (int x = 0; x < _dest.rows; x++) {
            DYN_PRAGMA(HLS LOOP TRIPCOUNT max = MAX_OUT_HEIGHT)
            for (int y = 0; y < (_dest.cols >> XF_BITSHIFT(NPC)); y++) {
                DYN_PRAGMA(HLS LOOP TRIPCOUNT max = MAX_OUT_WIDTH)
#pragma HLS PIPELINE
                r = _r.data[x * (_dest.cols >> XF_BITSHIFT(NPC)) + y];
                g = _g.data[x * (_dest.cols >> XF_BITSHIFT(NPC)) + y];
                b = _b.data[x * (_dest.cols >> XF_BITSHIFT(NPC)) + y];
                for (int i = 0; i < (1 << XF_BITSHIFT(NPC)); i++) {
#pragma HLS UNROLL
                    dout(23 + i * 24, i * 24) = (r(i * 8 + 7, i * 8),
                                                 g(i * 8 + 7, i * 8),
                                                 b(i * 8 + 7, i * 8));
                }
                _dest.data[x * (_dest.cols >> XF_BITSHIFT(NPC)) + y] = dout;
            }
        }
    }

    void resize_blur_rgb(ap_uint<AXI_WIDTH> *image_in,
                         ap_uint<AXI_WIDTH> *image_out,
                         int width_in,
                         int height_in,
                         int width_out,
                         int height_out,
                         float sigma)
    {
#pragma HLS INTERFACE m_axi port = image_in offset = slave bundle = image_in_gmem
#pragma HLS INTERFACE m_axi port = image_out offset = slave bundle = image_out_gmem
#pragma HLS INTERFACE s_axilite port = image_in bundle = control
#pragma HLS INTERFACE s_axilite port = image_out bundle = control
#pragma HLS INTERFACE s_axilite port = width_in bundle = control
#pragma HLS INTERFACE s_axilite port = height_in bundle = control
#pragma HLS INTERFACE s_axilite port = width_out bundle = control
#pragma HLS INTERFACE s_axilite port = height_out bundle = control
#pragma HLS INTERFACE s_axilite port = sigma bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

        xf::Mat<TYPE, MAX_IN_HEIGHT, MAX_IN_WIDTH, NPC> in_mat(height_in, width_in);
        DYN_PRAGMA(HLS stream variable = in_mat.data depth = STREAM_DEPTH)

        xf::Mat<XF_8UC1, MAX_IN_HEIGHT, MAX_IN_WIDTH, NPC> in_r(height_in, width_in);
        DYN_PRAGMA(HLS stream variable = in_r.data depth = STREAM_DEPTH)
        xf::Mat<XF_8UC1, MAX_IN_HEIGHT, MAX_IN_WIDTH, NPC> in_g(height_in, width_in);
        DYN_PRAGMA(HLS stream variable = in_g.data depth = STREAM_DEPTH)
        xf::Mat<XF_8UC1, MAX_IN_HEIGHT, MAX_IN_WIDTH, NPC> in_b(height_in, width_in);
        DYN_PRAGMA(HLS stream variable = in_b.data depth = STREAM_DEPTH)

        xf::Mat<XF_8UC1, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> out_r(height_out, width_out);
        DYN_PRAGMA(HLS stream variable = out_r.data depth = STREAM_DEPTH)
        xf::Mat<XF_8UC1, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> out_g(height_out, width_out);
        DYN_PRAGMA(HLS stream variable = out_g.data depth = STREAM_DEPTH)
        xf::Mat<XF_8UC1, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> out_b(height_out, width_out);
        DYN_PRAGMA(HLS stream variable = out_b.data depth = STREAM_DEPTH)

        xf::Mat<XF_8UC1, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> blur_r(height_out, width_out);
        DYN_PRAGMA(HLS stream variable = blur_r.data depth = STREAM_DEPTH)
        xf::Mat<XF_8UC1, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> blur_g(height_out, width_out);
        DYN_PRAGMA(HLS stream variable = blur_g.data depth = STREAM_DEPTH)
        xf::Mat<XF_8UC1, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> blur_b(height_out, width_out);
        DYN_PRAGMA(HLS stream variable = blur_b.data depth = STREAM_DEPTH)

        xf::Mat<XF_8UC3, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC> out_mat(height_out, width_out);
        DYN_PRAGMA(HLS stream variable = out_mat.data depth = STREAM_DEPTH)

#pragma HLS DATAFLOW

        xf::Array2xfMat<AXI_WIDTH, TYPE, MAX_IN_HEIGHT, MAX_IN_WIDTH, NPC>(image_in, in_mat);
        mat_split(in_mat, in_r, in_g, in_b);
        xf::resize<XF_INTERPOLATION_BILINEAR,
                   XF_8UC1,
                   MAX_IN_HEIGHT,
                   MAX_IN_WIDTH,
                   MAX_OUT_HEIGHT,
                   MAX_OUT_WIDTH,
                   NPC,
                   MAX_DOWN_SCALE>(in_r, blur_r);
        xf::resize<XF_INTERPOLATION_BILINEAR,
                   XF_8UC1,
                   MAX_IN_HEIGHT,
                   MAX_IN_WIDTH,
                   MAX_OUT_HEIGHT,
                   MAX_OUT_WIDTH,
                   NPC,
                   MAX_DOWN_SCALE>(in_g, blur_g);
        xf::resize<XF_INTERPOLATION_BILINEAR,
                   XF_8UC1,
                   MAX_IN_HEIGHT,
                   MAX_IN_WIDTH,
                   MAX_OUT_HEIGHT,
                   MAX_OUT_WIDTH,
                   NPC,
                   MAX_DOWN_SCALE>(in_b, blur_b);
        xf::GaussianBlur<FILTER_WIDTH,
                         XF_BORDER_CONSTANT,
                         XF_8UC1,
                         MAX_OUT_HEIGHT,
                         MAX_OUT_WIDTH,
                         NPC>(blur_r, out_r, sigma);
        xf::GaussianBlur<FILTER_WIDTH,
                         XF_BORDER_CONSTANT,
                         XF_8UC1,
                         MAX_OUT_HEIGHT,
                         MAX_OUT_WIDTH,
                         NPC>(blur_g, out_g, sigma);
        xf::GaussianBlur<FILTER_WIDTH,
                         XF_BORDER_CONSTANT,
                         XF_8UC1,
                         MAX_OUT_HEIGHT,
                         MAX_OUT_WIDTH,
                         NPC>(blur_b, out_b, sigma);
        mat_combine(out_r, out_g, out_b, out_mat);
        xf::xfMat2Array<AXI_WIDTH, TYPE, MAX_OUT_HEIGHT, MAX_OUT_WIDTH, NPC>(out_mat, image_out);
    }
}
