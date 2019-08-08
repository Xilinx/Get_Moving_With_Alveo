PLATFORM := /opt/xilinx/platforms/xilinx_u200_xdma_201830_2/xilinx_u200_xdma_201830_2.xpfm

XRT := $(SDKTARGETSYSROOT)/opt/xilinx/xrt

NCPUS := $(shell grep -c ^processor /proc/cpuinfo)
JOBS := $(shell expr $(NCPUS) - 1)

CXXFLAGS += -Wall -g -O0 -std=c++14 $(XRT_INC) -I$(XRT)/include -DDSA64 -DHAL2
CXXLDFLAGS += -I$(XRT)/include -L$(XRT)/lib -lxrt_core -ldl -luuid -pthread -lxilinxopencl -lOpenCL

RESIZE_FLAGS := --kernel resize_accel_rgb -I. -I./xfopencv/include --xp prop:kernel.resize_accel_rgb.kernel_flags="-std=c++0x -D__SDSVHLS__"
RESIZE_BLUR_FLAGS := --kernel resize_blur_rgb -I. -I./xfopencv/include --xp prop:kernel.resize_blur_rgb.kernel_flags="-std=c++0x -D__SDSVHLS__"

XOCCFLAGS := --platform $(PLATFORM) -t hw  -s -g --profile_kernel data:wide_vadd:all:all --profile_kernel data:resize_blur_rgb:all:all
XOCCLFLAGS := --jobs $(JOBS)
XOCCLFLAGS += --sp resize_accel_rgb_1.m_axi_image_in_gmem:DDR[0] --sp resize_accel_rgb_1.m_axi_image_out_gmem:DDR[0]
XOCCLFLAGS += --sp resize_blur_rgb_1.m_axi_image_in_gmem:DDR[2] --sp resize_blur_rgb_1.m_axi_image_out_gmem:DDR[1]
XOCCLFLAGS += --sp wide_vadd_1.m_axi_gmem:DDR[1] --sp wide_vadd_1.m_axi_gmem1:DDR[2] --sp wide_vadd_1.m_axi_gmem2:DDR[1]
XOCCLFLAGS += --slr vadd_1:SLR1 --slr wide_vadd_1:SLR1
XOCCLFLAGS += --slr resize_accel_rgb_1:SLR0 --slr resize_blur_rgb_1:SLR1

XOS = vadd.xo wide_vadd.xo resize_rgb.xo resize_blur.xo

.phony: clean traces

all: alveo_examples.xclbin

alveo_examples.xclbin: $(XOS)
	xocc -l $(XOCCFLAGS) $(XOCCLFLAGS) -o $@ $(XOS)

vadd.xo: vadd.cpp
	xocc --kernel vadd $(XOCCFLAGS) -c -o $@ $<

wide_vadd.xo: wide_vadd.cpp
	xocc --kernel wide_vadd $(XOCCFLAGS) -c -o $@ $<

resize_rgb.xo: resize_rgb.cpp
	xocc $(XOCCFLAGS) $(RESIZE_FLAGS) -c -o $@ $<

resize_blur.xo: resize_blur.cpp
	xocc $(XOCCFLAGS) $(RESIZE_BLUR_FLAGS) -c -o $@ $<

clean:
	$(RM) -r *.xo _x .Xil sd_card *.xclbin *.ltx *.log *.info
