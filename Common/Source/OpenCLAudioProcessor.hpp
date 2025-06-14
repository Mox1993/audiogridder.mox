/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * OpenCL Audio Processing Implementation
 */

#pragma once

#include "GPUAudioProcessor.hpp"

#ifdef AG_ENABLE_OPENCL
#include <CL/cl.h>
#include <memory>

namespace e47 {

class OpenCLAudioProcessor : public GPUAudioProcessor {
public:
    OpenCLAudioProcessor();
    ~OpenCLAudioProcessor() override;

    // Static methods
    static std::vector<GPUDeviceInfo> getAvailableDevices();
    static bool isAvailable();

    // Device management
    bool initialize(int deviceId = -1) override;
    void shutdown() override;
    bool isInitialized() const override;

    // Audio processing methods
    bool processAudioBuffer(AudioBuffer<float>& buffer, int numSamples) override;
    bool processAudioBuffer(AudioBuffer<double>& buffer, int numSamples) override;

    // Memory management
    bool allocateBuffers(int numChannels, int maxSamples) override;
    void deallocateBuffers() override;

    // Performance monitoring
    double getLastProcessingTimeMs() const override;
    size_t getMemoryUsage() const override;

    // Configuration
    void setProcessingPrecision(bool useDouble) override;
    bool supportsDoublePrecision() const override;

private:
    // OpenCL-specific members
    cl_platform_id m_platform;
    cl_device_id m_device;
    cl_context m_context;
    cl_command_queue m_queue;
    cl_program m_program;
    cl_kernel m_kernelFloat;
    cl_kernel m_kernelDouble;
    
    // Device memory buffers
    cl_mem m_deviceBufferFloat;
    cl_mem m_deviceBufferDouble;
    
    // Host memory for staging
    float* m_hostBufferFloat;
    double* m_hostBufferDouble;
    
    size_t m_bufferSizeBytes;
    bool m_buffersAllocated;
    
    // Performance tracking
    cl_event m_event;
    
    // Helper methods
    bool checkOpenCLError(cl_int error, const char* operation);
    bool createKernels();
    void copyToDevice(const AudioBuffer<float>& buffer, int numSamples);
    void copyToDevice(const AudioBuffer<double>& buffer, int numSamples);
    void copyFromDevice(AudioBuffer<float>& buffer, int numSamples);
    void copyFromDevice(AudioBuffer<double>& buffer, int numSamples);
    
    // Audio processing kernels
    bool processAudioKernel(cl_mem deviceData, int numChannels, int numSamples, bool useDouble);
    
    // OpenCL kernel source
    static const char* getKernelSource();
};

} // namespace e47

#endif // AG_ENABLE_OPENCL
