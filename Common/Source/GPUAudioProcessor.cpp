/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * GPU Audio Processing Infrastructure
 */

#include "GPUAudioProcessor.hpp"

#ifdef AG_ENABLE_CUDA
#include "CUDAAudioProcessor.hpp"
#endif

#ifdef AG_ENABLE_OPENCL
#include "OpenCLAudioProcessor.hpp"
#endif

namespace e47 {

GPUAudioProcessor::GPUAudioProcessor() = default;

GPUAudioProcessor::~GPUAudioProcessor() = default;

std::vector<GPUDeviceInfo> GPUAudioProcessor::getAvailableDevices() {
    std::vector<GPUDeviceInfo> devices;
    
#ifdef AG_ENABLE_CUDA
    auto cudaDevices = CUDAAudioProcessor::getAvailableDevices();
    devices.insert(devices.end(), cudaDevices.begin(), cudaDevices.end());
#endif

#ifdef AG_ENABLE_OPENCL
    auto openclDevices = OpenCLAudioProcessor::getAvailableDevices();
    devices.insert(devices.end(), openclDevices.begin(), openclDevices.end());
#endif

    return devices;
}

bool GPUAudioProcessor::isGPUAvailable() {
    auto devices = getAvailableDevices();
    return !devices.empty();
}

GPUBackend GPUAudioProcessor::getBestAvailableBackend() {
#ifdef AG_ENABLE_CUDA
    if (CUDAAudioProcessor::isAvailable()) {
        return GPUBackend::CUDA;
    }
#endif

#ifdef AG_ENABLE_OPENCL
    if (OpenCLAudioProcessor::isAvailable()) {
        return GPUBackend::OPENCL;
    }
#endif

    return GPUBackend::NONE;
}

std::unique_ptr<GPUAudioProcessor> GPUAudioProcessor::create(GPUBackend backend) {
    if (backend == GPUBackend::AUTO) {
        backend = getBestAvailableBackend();
    }
    
    switch (backend) {
#ifdef AG_ENABLE_CUDA
        case GPUBackend::CUDA:
            return std::make_unique<CUDAAudioProcessor>();
#endif

#ifdef AG_ENABLE_OPENCL
        case GPUBackend::OPENCL:
            return std::make_unique<OpenCLAudioProcessor>();
#endif

        default:
            return nullptr;
    }
}

} // namespace e47
