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

#include <iostream>
#include <string>

// Xilinx OpenCL and XRT includes
#include "xcl2.hpp"

#include <CL/cl.h>


int main(int argc, char *argv[])
{
    // Initialize an event timer we'll use for monitoring the application
    EventTimer et;

    if (argc != 2) {
        std::cout << "Usage: 00_load_kernels <xclbin>";
        return EXIT_FAILURE;
    }

    std::cout << "-- Example 0: Loading the FPGA Binary --" << std::endl
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
    cl::CommandQueue q(context, device);

    std::string device_name    = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile     = xcl::find_binary_file(device_name, argv[1]);
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);

    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program, "vadd");
    et.finish();

    std::cout << std::endl
              << "FPGA programmed, example complete!" << std::endl
              << std::endl;

    std::cout << "-- Key execution times --" << std::endl;

    et.print();
}
