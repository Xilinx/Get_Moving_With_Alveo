# UG1352: Get Moving with Alveo

This repository contains the source files for the exercises in *UG1352: Get Moving
With Alveo*. 

This repository majes use of Git submodules to pull in other repositories such as
the xf::OpenCV hardware accelerated library. To properly clone this repository
be sure to include the `--recurse-submodules` command line switch.

```bash
git clone --recurse-submodules https://github.com/Xilinx/xrt_onboarding.git
```

This repository includes both hardware and software sources.

## Building the Hardware Design

All hardware sources for this design are under the `hw_src` directory and can
be built by running `make`. Before doing so, ensure that the Makefile points
to your platform source location:

```Makefile
PLATFORM := /opt/xilinx/platforms/xilinx_u200_xdma_201830_2/xilinx_u200_xdma_201830_2.xpfm
```

If your platform is different or is installed in a different location ensure that
your Makefile is set appropriately. 

```bash
cd hw_src
make
```

## Building the Software Design

All software for these references is built using CMake. For some examples we rely
on external libraries, in particular OpenCV and OpenMP. If these libraries are not
present on your system the dependent examples will not be built.

From the root directory of this archive, to build the software:

```bash
mkdir build
cd build

cmake ..

make -j
```

**NOTE:** This should be done *after* building hardware so that the .xclbin file exists

To run any of the resulting examples, execute them directly with the xclbin file name
as per *UG1352: Get Moving With Alveo*.

```bash
./00_load_kernels alveo_examples
```
