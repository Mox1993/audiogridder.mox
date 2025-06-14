/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * CUDA Audio Processing Implementation
 */

#include "CUDAAudioProcessor.hpp"

#ifdef AG_ENABLE_CUDA

#include <iostream>
#include <chrono>

namespace e47 {

// CUDA kernel for basic audio processing (example: gain and filtering)
__global__ void processAudioKernelFloat(float* data, int numChannels, int numSamples, float gain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSamples = numChannels * numSamples;
    
    if (idx < totalSamples) {
        // Basic audio processing: apply gain and simple high-pass filter
        data[idx] *= gain;
        
        // Simple high-pass filter (example processing)
        if (idx > numChannels) {
            data[idx] = data[idx] - 0.95f * data[idx - numChannels];
        }
    }
}

__global__ void processAudioKernelDouble(double* data, int numChannels, int numSamples, double gain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSamples = numChannels * numSamples;
    
    if (idx < totalSamples) {
        // Basic audio processing: apply gain and simple high-pass filter
        data[idx] *= gain;
        
        // Simple high-pass filter (example processing)
        if (idx > numChannels) {
            data[idx] = data[idx] - 0.95 * data[idx - numChannels];
        }
    }
}

CUDAAudioProcessor::CUDAAudioProcessor()
    : m_stream(nullptr)
    , m_cublasHandle(nullptr)
    , m_deviceBufferFloat(nullptr)
    , m_deviceBufferDouble(nullptr)
    , m_hostBufferFloat(nullptr)
    , m_hostBufferDouble(nullptr)
    , m_bufferSizeBytes(0)
    , m_buffersAllocated(false)
    , m_startEvent(nullptr)
    , m_stopEvent(nullptr) {
}

CUDAAudioProcessor::~CUDAAudioProcessor() {
    shutdown();
}

std::vector<GPUDeviceInfo> CUDAAudioProcessor::getAvailableDevices() {
    std::vector<GPUDeviceInfo> devices;
    
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        return devices;
    }
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            GPUDeviceInfo info;
            info.deviceId = i;
            info.name = String(prop.name);
            info.totalMemory = prop.totalGlobalMem;
            
            size_t free, total;
            cudaSetDevice(i);
            if (cudaMemGetInfo(&free, &total) == cudaSuccess) {
                info.freeMemory = free;
            } else {
                info.freeMemory = 0;
            }
            
            info.computeCapability = prop.major * 10 + prop.minor;
            info.backend = GPUBackend::CUDA;
            info.isAvailable = prop.major >= 3; // Require compute capability 3.0+
            
            devices.push_back(info);
        }
    }
    
    return devices;
}

bool CUDAAudioProcessor::isAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

bool CUDAAudioProcessor::initialize(int deviceId) {
    if (m_initialized) {
        return true;
    }
    
    // Set device
    if (deviceId >= 0) {
        if (!checkCudaError(cudaSetDevice(deviceId), "cudaSetDevice")) {
            return false;
        }
        m_deviceId = deviceId;
    } else {
        // Use default device
        m_deviceId = 0;
    }
    
    // Create CUDA stream
    if (!checkCudaError(cudaStreamCreate(&m_stream), "cudaStreamCreate")) {
        return false;
    }
    
    // Create cuBLAS handle
    if (!checkCublasError(cublasCreate(&m_cublasHandle), "cublasCreate")) {
        return false;
    }
    
    if (!checkCublasError(cublasSetStream(m_cublasHandle, m_stream), "cublasSetStream")) {
        return false;
    }
    
    // Create events for timing
    if (!checkCudaError(cudaEventCreate(&m_startEvent), "cudaEventCreate start")) {
        return false;
    }
    
    if (!checkCudaError(cudaEventCreate(&m_stopEvent), "cudaEventCreate stop")) {
        return false;
    }
    
    m_initialized = true;
    return true;
}

void CUDAAudioProcessor::shutdown() {
    if (!m_initialized) {
        return;
    }
    
    deallocateBuffers();
    
    if (m_startEvent) {
        cudaEventDestroy(m_startEvent);
        m_startEvent = nullptr;
    }
    
    if (m_stopEvent) {
        cudaEventDestroy(m_stopEvent);
        m_stopEvent = nullptr;
    }
    
    if (m_cublasHandle) {
        cublasDestroy(m_cublasHandle);
        m_cublasHandle = nullptr;
    }
    
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    
    m_initialized = false;
}

bool CUDAAudioProcessor::isInitialized() const {
    return m_initialized;
}

bool CUDAAudioProcessor::processAudioBuffer(AudioBuffer<float>& buffer, int numSamples) {
    if (!m_initialized || !m_buffersAllocated) {
        return false;
    }
    
    cudaEventRecord(m_startEvent, m_stream);
    
    // Copy data to device
    copyToDevice(buffer, numSamples);
    
    // Process on GPU
    if (!processAudioKernel(m_deviceBufferFloat, buffer.getNumChannels(), numSamples)) {
        return false;
    }
    
    // Copy data back to host
    copyFromDevice(buffer, numSamples);
    
    cudaEventRecord(m_stopEvent, m_stream);
    cudaEventSynchronize(m_stopEvent);
    
    // Calculate processing time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    m_lastProcessingTime = static_cast<double>(milliseconds);
    
    return true;
}

bool CUDAAudioProcessor::processAudioBuffer(AudioBuffer<double>& buffer, int numSamples) {
    if (!m_initialized || !m_buffersAllocated || !m_useDoublePrecision) {
        return false;
    }
    
    cudaEventRecord(m_startEvent, m_stream);
    
    // Copy data to device
    copyToDevice(buffer, numSamples);
    
    // Process on GPU
    if (!processAudioKernel(m_deviceBufferDouble, buffer.getNumChannels(), numSamples)) {
        return false;
    }
    
    // Copy data back to host
    copyFromDevice(buffer, numSamples);
    
    cudaEventRecord(m_stopEvent, m_stream);
    cudaEventSynchronize(m_stopEvent);
    
    // Calculate processing time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    m_lastProcessingTime = static_cast<double>(milliseconds);
    
    return true;
}

bool CUDAAudioProcessor::allocateBuffers(int numChannels, int maxSamples) {
    if (m_buffersAllocated) {
        deallocateBuffers();
    }
    
    m_numChannels = numChannels;
    m_maxSamples = maxSamples;
    m_bufferSizeBytes = numChannels * maxSamples * sizeof(float);
    
    // Allocate device memory for float
    if (!checkCudaError(cudaMalloc(&m_deviceBufferFloat, m_bufferSizeBytes), "cudaMalloc float")) {
        return false;
    }
    
    // Allocate host pinned memory for float
    if (!checkCudaError(cudaMallocHost(&m_hostBufferFloat, m_bufferSizeBytes), "cudaMallocHost float")) {
        return false;
    }
    
    if (m_useDoublePrecision) {
        size_t doubleBufSize = numChannels * maxSamples * sizeof(double);
        
        // Allocate device memory for double
        if (!checkCudaError(cudaMalloc(&m_deviceBufferDouble, doubleBufSize), "cudaMalloc double")) {
            return false;
        }
        
        // Allocate host pinned memory for double
        if (!checkCudaError(cudaMallocHost(&m_hostBufferDouble, doubleBufSize), "cudaMallocHost double")) {
            return false;
        }
    }
    
    m_buffersAllocated = true;
    return true;
}

void CUDAAudioProcessor::deallocateBuffers() {
    if (!m_buffersAllocated) {
        return;
    }
    
    if (m_deviceBufferFloat) {
        cudaFree(m_deviceBufferFloat);
        m_deviceBufferFloat = nullptr;
    }
    
    if (m_hostBufferFloat) {
        cudaFreeHost(m_hostBufferFloat);
        m_hostBufferFloat = nullptr;
    }
    
    if (m_deviceBufferDouble) {
        cudaFree(m_deviceBufferDouble);
        m_deviceBufferDouble = nullptr;
    }
    
    if (m_hostBufferDouble) {
        cudaFreeHost(m_hostBufferDouble);
        m_hostBufferDouble = nullptr;
    }
    
    m_buffersAllocated = false;
}

double CUDAAudioProcessor::getLastProcessingTimeMs() const {
    return m_lastProcessingTime;
}

size_t CUDAAudioProcessor::getMemoryUsage() const {
    if (!m_buffersAllocated) {
        return 0;
    }
    
    size_t usage = m_bufferSizeBytes * 2; // Device + host float buffers
    
    if (m_useDoublePrecision) {
        usage += m_numChannels * m_maxSamples * sizeof(double) * 2; // Device + host double buffers
    }
    
    return usage;
}

void CUDAAudioProcessor::setProcessingPrecision(bool useDouble) {
    m_useDoublePrecision = useDouble;
}

bool CUDAAudioProcessor::supportsDoublePrecision() const {
    return true; // CUDA supports double precision
}

bool CUDAAudioProcessor::checkCudaError(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

bool CUDAAudioProcessor::checkCublasError(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error in " << operation << ": " << status << std::endl;
        return false;
    }
    return true;
}

void CUDAAudioProcessor::copyToDevice(const AudioBuffer<float>& buffer, int numSamples) {
    // Interleave audio data for GPU processing
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        const float* channelData = buffer.getReadPointer(ch);
        for (int sample = 0; sample < numSamples; ++sample) {
            m_hostBufferFloat[sample * buffer.getNumChannels() + ch] = channelData[sample];
        }
    }
    
    size_t copySize = buffer.getNumChannels() * numSamples * sizeof(float);
    cudaMemcpyAsync(m_deviceBufferFloat, m_hostBufferFloat, copySize, cudaMemcpyHostToDevice, m_stream);
}

void CUDAAudioProcessor::copyToDevice(const AudioBuffer<double>& buffer, int numSamples) {
    // Interleave audio data for GPU processing
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        const double* channelData = buffer.getReadPointer(ch);
        for (int sample = 0; sample < numSamples; ++sample) {
            m_hostBufferDouble[sample * buffer.getNumChannels() + ch] = channelData[sample];
        }
    }
    
    size_t copySize = buffer.getNumChannels() * numSamples * sizeof(double);
    cudaMemcpyAsync(m_deviceBufferDouble, m_hostBufferDouble, copySize, cudaMemcpyHostToDevice, m_stream);
}

void CUDAAudioProcessor::copyFromDevice(AudioBuffer<float>& buffer, int numSamples) {
    size_t copySize = buffer.getNumChannels() * numSamples * sizeof(float);
    cudaMemcpyAsync(m_hostBufferFloat, m_deviceBufferFloat, copySize, cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
    
    // De-interleave audio data
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        float* channelData = buffer.getWritePointer(ch);
        for (int sample = 0; sample < numSamples; ++sample) {
            channelData[sample] = m_hostBufferFloat[sample * buffer.getNumChannels() + ch];
        }
    }
}

void CUDAAudioProcessor::copyFromDevice(AudioBuffer<double>& buffer, int numSamples) {
    size_t copySize = buffer.getNumChannels() * numSamples * sizeof(double);
    cudaMemcpyAsync(m_hostBufferDouble, m_deviceBufferDouble, copySize, cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
    
    // De-interleave audio data
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        double* channelData = buffer.getWritePointer(ch);
        for (int sample = 0; sample < numSamples; ++sample) {
            channelData[sample] = m_hostBufferDouble[sample * buffer.getNumChannels() + ch];
        }
    }
}

bool CUDAAudioProcessor::processAudioKernel(float* deviceData, int numChannels, int numSamples) {
    int totalSamples = numChannels * numSamples;
    int blockSize = 256;
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    
    // Example processing: apply gain of 1.0 (can be made configurable)
    processAudioKernelFloat<<<gridSize, blockSize, 0, m_stream>>>(deviceData, numChannels, numSamples, 1.0f);
    
    return checkCudaError(cudaGetLastError(), "processAudioKernelFloat");
}

bool CUDAAudioProcessor::processAudioKernel(double* deviceData, int numChannels, int numSamples) {
    int totalSamples = numChannels * numSamples;
    int blockSize = 256;
    int gridSize = (totalSamples + blockSize - 1) / blockSize;
    
    // Example processing: apply gain of 1.0 (can be made configurable)
    processAudioKernelDouble<<<gridSize, blockSize, 0, m_stream>>>(deviceData, numChannels, numSamples, 1.0);
    
    return checkCudaError(cudaGetLastError(), "processAudioKernelDouble");
}

} // namespace e47

#endif // AG_ENABLE_CUDA
