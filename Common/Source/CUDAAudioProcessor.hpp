/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * CUDA Audio Processing Implementation
 */

#pragma once

#include "GPUAudioProcessor.hpp"

#ifdef AG_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>

namespace e47 {

class CUDAAudioProcessor : public GPUAudioProcessor {
public:
    CUDAAudioProcessor();
    ~CUDAAudioProcessor() override;

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
    // CUDA-specific members
    cudaStream_t m_stream;
    cublasHandle_t m_cublasHandle;
    
    // Device memory pointers
    float* m_deviceBufferFloat;
    double* m_deviceBufferDouble;
    
    // Host memory for staging
    float* m_hostBufferFloat;
    double* m_hostBufferDouble;
    
    size_t m_bufferSizeBytes;
    bool m_buffersAllocated;
    
    // Performance tracking
    cudaEvent_t m_startEvent;
    cudaEvent_t m_stopEvent;
    
    // Helper methods
    bool checkCudaError(cudaError_t error, const char* operation);
    bool checkCublasError(cublasStatus_t status, const char* operation);
    void copyToDevice(const AudioBuffer<float>& buffer, int numSamples);
    void copyToDevice(const AudioBuffer<double>& buffer, int numSamples);
    void copyFromDevice(AudioBuffer<float>& buffer, int numSamples);
    void copyFromDevice(AudioBuffer<double>& buffer, int numSamples);
    
    // Audio processing kernels
    bool processAudioKernel(float* deviceData, int numChannels, int numSamples);
    bool processAudioKernel(double* deviceData, int numChannels, int numSamples);
};

} // namespace e47

#endif // AG_ENABLE_CUDA
