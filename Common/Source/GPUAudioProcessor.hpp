/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * GPU Audio Processing Infrastructure
 */

#pragma once

#include <memory>
#include <vector>
#include "JuceHeader.h"

namespace e47 {

enum class GPUBackend {
    NONE = 0,
    CUDA,
    OPENCL,
    AUTO
};

struct GPUDeviceInfo {
    int deviceId;
    String name;
    size_t totalMemory;
    size_t freeMemory;
    int computeCapability;
    GPUBackend backend;
    bool isAvailable;
};

class GPUAudioProcessor {
public:
    GPUAudioProcessor();
    virtual ~GPUAudioProcessor();

    // Static methods for GPU detection and management
    static std::vector<GPUDeviceInfo> getAvailableDevices();
    static bool isGPUAvailable();
    static GPUBackend getBestAvailableBackend();
    
    // Device management
    virtual bool initialize(int deviceId = -1) = 0;
    virtual void shutdown() = 0;
    virtual bool isInitialized() const = 0;
    
    // Audio processing methods
    virtual bool processAudioBuffer(AudioBuffer<float>& buffer, int numSamples) = 0;
    virtual bool processAudioBuffer(AudioBuffer<double>& buffer, int numSamples) = 0;
    
    // Memory management
    virtual bool allocateBuffers(int numChannels, int maxSamples) = 0;
    virtual void deallocateBuffers() = 0;
    
    // Performance monitoring
    virtual double getLastProcessingTimeMs() const = 0;
    virtual size_t getMemoryUsage() const = 0;
    
    // Configuration
    virtual void setProcessingPrecision(bool useDouble) = 0;
    virtual bool supportsDoublePrecision() const = 0;
    
    // Factory method
    static std::unique_ptr<GPUAudioProcessor> create(GPUBackend backend);

protected:
    bool m_initialized = false;
    bool m_useDoublePrecision = false;
    int m_deviceId = -1;
    int m_numChannels = 0;
    int m_maxSamples = 0;
    double m_lastProcessingTime = 0.0;
};

} // namespace e47
