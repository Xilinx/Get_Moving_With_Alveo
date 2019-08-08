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


#include "event_timer.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <string>

// Xilinx OpenCL and XRT includes
#include "xcl2.hpp"

#include <CL/cl.h>

#define BUFSIZE (1024 * 1024 * 256)
#define NUM_BUFS 10

void vadd_sw(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t size)
{
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

int subdivide_buffer(std::vector<cl::Buffer> &divided,
                     cl::Buffer buf_in,
                     cl_mem_flags flags,
                     int num_divisions)
{
    // Get the size of the buffer
    size_t size;
    size = buf_in.getInfo<CL_MEM_SIZE>();

    if (size / num_divisions <= 4096) {
        return -1;
    }

    cl_buffer_region region;

    int err;
    region.origin = 0;
    region.size   = size / num_divisions;

    // Round region size up to nearest 4k for efficient burst behavior
    if (region.size % 4096 != 0) {
        region.size += (4096 - (region.size % 4096));
    }

    for (int i = 0; i < num_divisions; i++) {
        if (i == num_divisions - 1) {
            if ((region.origin + region.size) > size) {
                region.size = size - region.origin;
            }
        }
        cl::Buffer buf = buf_in.createSubBuffer(flags,
                                                CL_BUFFER_CREATE_TYPE_REGION,
                                                &region,
                                                &err);
        if (err != CL_SUCCESS) {
            return err;
        }
        divided.push_back(buf);
        region.origin += region.size;
    }

    return 0;
}

int enqueue_subbuf_vadd(cl::CommandQueue &q,
                        cl::Kernel &krnl,
                        cl::Event &event,
                        cl::Buffer a,
                        cl::Buffer b,
                        cl::Buffer c)
{
    // Get the size of the buffer
    cl::Event k_event, m_event;
    std::vector<cl::Event> krnl_events;

    static std::vector<cl::Event> tx_events, rx_events;

    std::vector<cl::Memory> c_vec;
    size_t size;
    size = a.getInfo<CL_MEM_SIZE>();

    std::vector<cl::Memory> in_vec;
    in_vec.push_back(a);
    in_vec.push_back(b);
    q.enqueueMigrateMemObjects(in_vec, 0, &tx_events, &m_event);
    krnl_events.push_back(m_event);
    tx_events.push_back(m_event);
    if (tx_events.size() > 1) {
        tx_events[0] = tx_events[1];
        tx_events.pop_back();
    }

    krnl.setArg(0, a);
    krnl.setArg(1, b);
    krnl.setArg(2, c);
    krnl.setArg(3, (uint32_t)(size / sizeof(uint32_t)));

    q.enqueueTask(krnl, &krnl_events, &k_event);
    krnl_events.push_back(k_event);
    if (rx_events.size() == 1) {
        krnl_events.push_back(rx_events[0]);
        rx_events.pop_back();
    }
    c_vec.push_back(c);
    q.enqueueMigrateMemObjects(c_vec,
                               CL_MIGRATE_MEM_OBJECT_HOST,
                               &krnl_events,
                               &event);
    rx_events.push_back(event);

    return 0;
}

int main(int argc, char *argv[])
{
    // Initialize an event timer we'll use for monitoring the application
    EventTimer et;

    if (argc != 2) {
        std::cout << "Usage: 05_pipelined_vadd <xclbin>";
        return EXIT_FAILURE;
    }

    std::cout << "-- Example 5: Pipelining Kernel Execution --" << std::endl
              << std::endl;

    // Initialize the runtime (including a command queue) and load the
    // FPGA image
    std::cout << "Loading XCLBin to program the Alveo board:" << std::endl
              << std::endl;
    et.add("OpenCL Initialization");

    // This application will use the first Xilinx device found in the system
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device               = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    std::string device_name    = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile     = xcl::find_binary_file(device_name, argv[1]);
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program, "wide_vadd");
    et.finish();

    /// New code for example 01
    std::cout << std::endl
              << std::endl;
    std::cout << " -- Running kernel test with XRT-allocated contiguous buffers" << std::endl;
    std::cout << "    and wide VADD (16 values/clock)" << std::endl;

    // Map our user-allocated buffers as OpenCL buffers using a shared
    // host pointer
    try {
        et.add("Allocate contiguous OpenCL buffers");
        std::vector<cl::Memory> inBufVec;
        cl_mem_ext_ptr_t bank1_ext, bank2_ext;
        bank2_ext.flags = 2 | XCL_MEM_TOPOLOGY;
        bank2_ext.obj   = NULL;
        bank2_ext.param = 0;
        bank1_ext.flags = 1 | XCL_MEM_TOPOLOGY;
        bank1_ext.obj   = NULL;
        bank1_ext.param = 0;
        cl::Buffer a_buf(context,
                         static_cast<cl_mem_flags>(CL_MEM_READ_ONLY |
                                                   CL_MEM_EXT_PTR_XILINX),
                         BUFSIZE * sizeof(uint32_t),
                         &bank1_ext,
                         NULL);
        cl::Buffer b_buf(context,
                         static_cast<cl_mem_flags>(CL_MEM_READ_ONLY |
                                                   CL_MEM_EXT_PTR_XILINX),
                         BUFSIZE * sizeof(uint32_t),
                         &bank2_ext,
                         NULL);
        cl::Buffer c_buf(context,
                         static_cast<cl_mem_flags>(CL_MEM_READ_WRITE |
                                                   CL_MEM_EXT_PTR_XILINX),
                         BUFSIZE * sizeof(uint32_t),
                         &bank1_ext,
                         NULL);
        uint32_t *d = new uint32_t[BUFSIZE];
        et.finish();

        et.add("Map buffers to userspace pointers");
        uint32_t *a = (uint32_t *)q.enqueueMapBuffer(a_buf,
                                                     CL_TRUE,
                                                     CL_MAP_WRITE,
                                                     0,
                                                     BUFSIZE * sizeof(uint32_t));
        uint32_t *b = (uint32_t *)q.enqueueMapBuffer(b_buf,
                                                     CL_TRUE,
                                                     CL_MAP_WRITE,
                                                     0,
                                                     BUFSIZE * sizeof(uint32_t));
        uint32_t *c = (uint32_t *)q.enqueueMapBuffer(c_buf,
                                                     CL_TRUE,
                                                     CL_MAP_READ,
                                                     0,
                                                     BUFSIZE * sizeof(uint32_t));
        et.finish();


        et.add("Populating buffer inputs");
        for (int i = 0; i < BUFSIZE; i++) {
            a[i] = i;
            b[i] = 2 * i;
        }
        et.finish();

        // For comparison, let's have the CPU calculate the result
        et.add("Software VADD run");
        vadd_sw(a, b, d, BUFSIZE);
        et.finish();

        // Send the buffers down to the Alveo card
        et.add("Memory object migration enqueue");
        q.enqueueUnmapMemObject(a_buf, a);
        q.enqueueUnmapMemObject(b_buf, b);

        std::vector<cl::Buffer> a_bufs, b_bufs, c_bufs;
        subdivide_buffer(a_bufs, a_buf, CL_MEM_READ_ONLY, NUM_BUFS);
        subdivide_buffer(b_bufs, b_buf, CL_MEM_READ_ONLY, NUM_BUFS);
        subdivide_buffer(c_bufs, c_buf, CL_MEM_WRITE_ONLY, NUM_BUFS);

        std::array<cl::Event, NUM_BUFS> kernel_events;

        for (int i = 0; i < NUM_BUFS; i++) {
            enqueue_subbuf_vadd(q, krnl, kernel_events[i], a_bufs[i], b_bufs[i], c_bufs[i]);
        }

        et.add("Wait for kernels to complete");
        clWaitForEvents(NUM_BUFS, (const cl_event *)&kernel_events);

        et.finish();


        // Verify the results
        bool verified = true;
        for (int i = 0; i < BUFSIZE; i++) {
            if (c[i] != d[i]) {
                verified = false;
                std::cout << "ERROR: software and hardware vadd do not match: "
                          << c[i] << "!=" << d[i] << " at position " << i << std::endl;
                break;
            }
        }

        if (verified) {
            std::cout
                << std::endl
                << "OCL-mapped contiguous buffer example complete!"
                << std::endl
                << std::endl;
        }
        else {
            std::cout
                << std::endl
                << "OCL-mapped contiguous buffer example complete! (with errors)"
                << std::endl
                << std::endl;
        }

        std::cout << "--------------- Key execution times ---------------" << std::endl;


        q.enqueueUnmapMemObject(c_buf, c);
        free(d);
        q.finish();


        et.print();
    }
    catch (cl::Error &e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
