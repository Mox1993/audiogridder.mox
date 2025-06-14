/*
 * Copyright (c) 2024 AudioGridder GPU Extension
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * OpenCL Audio Processing Implementation
 */

#include "OpenCLAudioProcessor.hpp"

#ifdef AG_ENABLE_OPENCL

#include <iostream>
#include <chrono>
#include <cstring>

namespace e47 {

const char* OpenCLAudioProcessor::getKernelSource() {
    return R"(
__kernel void processAudioFloat(__global float* data, int numChannels, int numSamples, float gain) {
    int idx = get_global_id(0);
    int totalSamples = numChannels * numSamples;
    
    if (idx < totalSamples) {
        // Basic audio processing: apply gain and simple high-pass filter
        data[idx] *= gain;
        
        // Simple high-pass filter (example processing)
        if (idx >= numChannels) {
            data[idx] = data[idx] - 0.95f * data[idx - numChannels];
        }
    }
}

__kernel void processAudioDouble(__global double* data, int numChannels, int numSamples, double gain) {
    int idx = get_global_id(0);
    int totalSamples = numChannels * numSamples;
    
    if (idx < totalSamples) {
        // Basic audio processing: apply gain and simple high-pass filter
        data[idx] *= gain;
        
        // Simple high-pass filter (example processing)
        if (idx >= numChannels) {
            data[idx] = data[idx] - 0.95 * data[idx - numChannels];
        }
    }
}
)";
}

OpenCLAudioProcessor::OpenCLAudioProcessor()
    : m_platform(nullptr)
    , m_device(nullptr)
    , m_context(nullptr)
    , m_queue(nullptr)
    , m_program(nullptr)
    , m_kernelFloat(nullptr)
    , m_kernelDouble(nullptr)
    , m_deviceBufferFloat(nullptr)
    , m_deviceBufferDouble(nullptr)
    , m_hostBufferFloat(nullptr)
    , m_hostBufferDouble(nullptr)
    , m_bufferSizeBytes(0)
    , m_buffersAllocated(false)
    , m_event(nullptr) {
}

OpenCLAudioProcessor::~OpenCLAudioProcessor() {
    shutdown();
}

std::vector<GPUDeviceInfo> OpenCLAudioProcessor::getAvailableDevices() {
    std::vector<GPUDeviceInfo> devices;
    
    cl_uint numPlatforms = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (error != CL_SUCCESS || numPlatforms == 0) {
        return devices;
    }
    
    std::vector<cl_platform_id> platforms(numPlatforms);
    error = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (error != CL_SUCCESS) {
        return devices;
    }
    
    int deviceIdCounter = 0;
    for (const auto& platform : platforms) {
        cl_uint numDevices = 0;
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (error != CL_SUCCESS || numDevices == 0) {
            continue;
        }
        
        std::vector<cl_device_id> platformDevices(numDevices);
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, platformDevices.data(), nullptr);
        if (error != CL_SUCCESS) {
            continue;
        }
        
        for (const auto& device : platformDevices) {
            GPUDeviceInfo info;
            info.deviceId = deviceIdCounter++;
            info.backend = GPUBackend::OPENCL;
            
            // Get device name
            size_t nameSize = 0;
            clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &nameSize);
            if (nameSize > 0) {
                std::vector<char> name(nameSize);
                clGetDeviceInfo(device, CL_DEVICE_NAME, nameSize, name.data(), nullptr);
                info.name = String(name.data());
            }
            
            // Get memory info
            cl_ulong totalMem = 0;
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(totalMem), &totalMem, nullptr);
            info.totalMemory = static_cast<size_t>(totalMem);
            info.freeMemory = info.totalMemory; // OpenCL doesn't provide free memory directly
            
            // Check compute capability (simplified)
            cl_uint computeUnits = 0;
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
            info.computeCapability = static_cast<int>(computeUnits);
            
            info.isAvailable = true;
            devices.push_back(info);
        }
    }
    
    return devices;
}

bool OpenCLAudioProcessor::isAvailable() {
    cl_uint numPlatforms = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &numPlatforms);
    return (error == CL_SUCCESS && numPlatforms > 0);
}

bool OpenCLAudioProcessor::initialize(int deviceId) {
    if (m_initialized) {
        return true;
    }
    
    // Get platforms
    cl_uint numPlatforms = 0;
    cl_int error = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (!checkOpenCLError(error, "clGetPlatformIDs") || numPlatforms == 0) {
        return false;
    }
    
    std::vector<cl_platform_id> platforms(numPlatforms);
    error = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (!checkOpenCLError(error, "clGetPlatformIDs")) {
        return false;
    }
    
    // Find device
    bool deviceFound = false;
    int currentDeviceId = 0;
    
    for (const auto& platform : platforms) {
        cl_uint numDevices = 0;
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (error != CL_SUCCESS || numDevices == 0) {
            continue;
        }
        
        std::vector<cl_device_id> devices(numDevices);
        error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
        if (error != CL_SUCCESS) {
            continue;
        }
        
        for (const auto& device : devices) {
            if (deviceId < 0 || currentDeviceId == deviceId) {
                m_platform = platform;
                m_device = device;
                m_deviceId = currentDeviceId;
                deviceFound = true;
                break;
            }
            currentDeviceId++;
        }
        
        if (deviceFound) {
            break;
        }
    }
    
    if (!deviceFound) {
        std::cerr << "OpenCL device not found" << std::endl;
        return false;
    }
    
    // Create context
    m_context = clCreateContext(nullptr, 1, &m_device, nullptr, nullptr, &error);
    if (!checkOpenCLError(error, "clCreateContext")) {
        return false;
    }
    
    // Create command queue
    m_queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &error);
    if (!checkOpenCLError(error, "clCreateCommandQueue")) {
        return false;
    }
    
    // Create and build program
    const char* kernelSource = getKernelSource();
    m_program = clCreateProgramWithSource(m_context, 1, &kernelSource, nullptr, &error);
    if (!checkOpenCLError(error, "clCreateProgramWithSource")) {
        return false;
    }
    
    error = clBuildProgram(m_program, 1, &m_device, nullptr, nullptr, nullptr);
    if (error != CL_SUCCESS) {
        size_t logSize = 0;
        clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        if (logSize > 0) {
            std::vector<char> log(logSize);
            clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            std::cerr << "OpenCL build error: " << log.data() << std::endl;
        }
        return false;
    }
    
    // Create kernels
    if (!createKernels()) {
        return false;
    }
    
    m_initialized = true;
    return true;
}

void OpenCLAudioProcessor::shutdown() {
    if (!m_initialized) {
        return;
    }
    
    deallocateBuffers();
    
    if (m_kernelFloat) {
        clReleaseKernel(m_kernelFloat);
        m_kernelFloat = nullptr;
    }
    
    if (m_kernelDouble) {
        clReleaseKernel(m_kernelDouble);
        m_kernelDouble = nullptr;
    }
    
    if (m_program) {
        clReleaseProgram(m_program);
        m_program = nullptr;
    }
    
    if (m_queue) {
        clReleaseCommandQueue(m_queue);
        m_queue = nullptr;
    }
    
    if (m_context) {
        clReleaseContext(m_context);
        m_context = nullptr;
    }
    
    m_initialized = false;
}

bool OpenCLAudioProcessor::isInitialized() const {
    return m_initialized;
}

bool OpenCLAudioProcessor::processAudioBuffer(AudioBuffer<float>& buffer, int numSamples) {
    if (!m_initialized || !m_buffersAllocated) {
        return false;
    }
    
    // Copy data to device
    copyToDevice(buffer, numSamples);
    
    // Process on GPU
    if (!processAudioKernel(m_deviceBufferFloat, buffer.getNumChannels(), numSamples, false)) {
        return false;
    }
    
    // Copy data back to host
    copyFromDevice(buffer, numSamples);
    
    return true;
}

bool OpenCLAudioProcessor::processAudioBuffer(AudioBuffer<double>& buffer, int numSamples) {
    if (!m_initialized || !m_buffersAllocated || !m_useDoublePrecision) {
        return false;
    }
    
    // Copy data to device
    copyToDevice(buffer, numSamples);
    
    // Process on GPU
    if (!processAudioKernel(m_deviceBufferDouble, buffer.getNumChannels(), numSamples, true)) {
        return false;
    }
    
    // Copy data back to host
    copyFromDevice(buffer, numSamples);
    
    return true;
}

bool OpenCLAudioProcessor::allocateBuffers(int numChannels, int maxSamples) {
    if (m_buffersAllocated) {
        deallocateBuffers();
    }
    
    m_numChannels = numChannels;
    m_maxSamples = maxSamples;
    m_bufferSizeBytes = numChannels * maxSamples * sizeof(float);
    
    cl_int error;
    
    // Allocate device memory for float
    m_deviceBufferFloat = clCreateBuffer(m_context, CL_MEM_READ_WRITE, m_bufferSizeBytes, nullptr, &error);
    if (!checkOpenCLError(error, "clCreateBuffer float")) {
        return false;
    }
    
    // Allocate host memory for float
    m_hostBufferFloat = new float[numChannels * maxSamples];
    
    if (m_useDoublePrecision) {
        size_t doubleBufSize = numChannels * maxSamples * sizeof(double);
        
        // Allocate device memory for double
        m_deviceBufferDouble = clCreateBuffer(m_context, CL_MEM_READ_WRITE, doubleBufSize, nullptr, &error);
        if (!checkOpenCLError(error, "clCreateBuffer double")) {
            return false;
        }
        
        // Allocate host memory for double
        m_hostBufferDouble = new double[numChannels * maxSamples];
    }
    
    m_buffersAllocated = true;
    return true;
}

void OpenCLAudioProcessor::deallocateBuffers() {
    if (!m_buffersAllocated) {
        return;
    }
    
    if (m_deviceBufferFloat) {
        clReleaseMemObject(m_deviceBufferFloat);
        m_deviceBufferFloat = nullptr;
    }
    
    if (m_hostBufferFloat) {
        delete[] m_hostBufferFloat;
        m_hostBufferFloat = nullptr;
    }
    
    if (m_deviceBufferDouble) {
        clReleaseMemObject(m_deviceBufferDouble);
        m_deviceBufferDouble = nullptr;
    }
    
    if (m_hostBufferDouble) {
        delete[] m_hostBufferDouble;
        m_hostBufferDouble = nullptr;
    }
    
    m_buffersAllocated = false;
}

double OpenCLAudioProcessor::getLastProcessingTimeMs() const {
    return m_lastProcessingTime;
}

size_t OpenCLAudioProcessor::getMemoryUsage() const {
    if (!m_buffersAllocated) {
        return 0;
    }
    
    size_t usage = m_bufferSizeBytes * 2; // Device + host float buffers
    
    if (m_useDoublePrecision) {
        usage += m_numChannels * m_maxSamples * sizeof(double) * 2; // Device + host double buffers
    }
    
    return usage;
}

void OpenCLAudioProcessor::setProcessingPrecision(bool useDouble) {
    m_useDoublePrecision = useDouble;
}

bool OpenCLAudioProcessor::supportsDoublePrecision() const {
    if (!m_initialized) {
        return false;
    }
    
    cl_device_fp_config fpConfig = 0;
    cl_int error = clGetDeviceInfo(m_device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fpConfig), &fpConfig, nullptr);
    return (error == CL_SUCCESS && fpConfig != 0);
}

bool OpenCLAudioProcessor::checkOpenCLError(cl_int error, const char* operation) {
    if (error != CL_SUCCESS) {
        std::cerr << "OpenCL error in " << operation << ": " << error << std::endl;
        return false;
    }
    return true;
}

bool OpenCLAudioProcessor::createKernels() {
    cl_int error;
    
    // Create float kernel
    m_kernelFloat = clCreateKernel(m_program, "processAudioFloat", &error);
    if (!checkOpenCLError(error, "clCreateKernel float")) {
        return false;
    }
    
    // Create double kernel
    m_kernelDouble = clCreateKernel(m_program, "processAudioDouble", &error);
    if (!checkOpenCLError(error, "clCreateKernel double")) {
        return false;
    }
    
    return true;
}

void OpenCLAudioProcessor::copyToDevice(const AudioBuffer<float>& buffer, int numSamples) {
    // Interleave audio data for GPU processing
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        const float* channelData = buffer.getReadPointer(ch);
        for (int sample = 0; sample < numSamples; ++sample) {
            m_hostBufferFloat[sample * buffer.getNumChannels() + ch] = channelData[sample];
        }
    }
    
    size_t copySize = buffer.getNumChannels() * numSamples * sizeof(float);
    clEnqueueWriteBuffer(m_queue, m_deviceBufferFloat, CL_FALSE, 0, copySize, m_hostBufferFloat, 0, nullptr, nullptr);
}

void OpenCLAudioProcessor::copyToDevice(const AudioBuffer<double>& buffer, int numSamples) {
    // Interleave audio data for GPU processing
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        const double* channelData = buffer.getReadPointer(ch);
        for (int sample = 0; sample < numSamples; ++sample) {
            m_hostBufferDouble[sample * buffer.getNumChannels() + ch] = channelData[sample];
        }
    }
    
    size_t copySize = buffer.getNumChannels() * numSamples * sizeof(double);
    clEnqueueWriteBuffer(m_queue, m_deviceBufferDouble, CL_FALSE, 0, copySize, m_hostBufferDouble, 0, nullptr, nullptr);
}

void OpenCLAudioProcessor::copyFromDevice(AudioBuffer<float>& buffer, int numSamples) {
    size_t copySize = buffer.getNumChannels() * numSamples * sizeof(float);
    clEnqueueReadBuffer(m_queue, m_deviceBufferFloat, CL_TRUE, 0, copySize, m_hostBufferFloat, 0, nullptr, nullptr);
    
    // De-interleave audio data
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        float* channelData = buffer.getWritePointer(ch);
        for (int sample = 0; sample < numSamples; ++sample) {
            channelData[sample] = m_hostBufferFloat[sample * buffer.getNumChannels() + ch];
        }
    }
}

void OpenCLAudioProcessor::copyFromDevice(AudioBuffer<double>& buffer, int numSamples) {
    size_t copySize = buffer.getNumChannels() * numSamples * sizeof(double);
    clEnqueueReadBuffer(m_queue, m_deviceBufferDouble, CL_TRUE, 0, copySize, m_hostBufferDouble, 0, nullptr, nullptr);
    
    // De-interleave audio data
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        double* channelData = buffer.getWritePointer(ch);
        for (int sample = 0; sample < numSamples; ++sample) {
            channelData[sample] = m_hostBufferDouble[sample * buffer.getNumChannels() + ch];
        }
    }
}

bool OpenCLAudioProcessor::processAudioKernel(cl_mem deviceData, int numChannels, int numSamples, bool useDouble) {
    cl_int error;
    cl_kernel kernel = useDouble ? m_kernelDouble : m_kernelFloat;
    
    // Set kernel arguments
    error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &deviceData);
    if (!checkOpenCLError(error, "clSetKernelArg 0")) return false;
    
    error = clSetKernelArg(kernel, 1, sizeof(int), &numChannels);
    if (!checkOpenCLError(error, "clSetKernelArg 1")) return false;
    
    error = clSetKernelArg(kernel, 2, sizeof(int), &numSamples);
    if (!checkOpenCLError(error, "clSetKernelArg 2")) return false;
    
    if (useDouble) {
        double gain = 1.0;
        error = clSetKernelArg(kernel, 3, sizeof(double), &gain);
    } else {
        float gain = 1.0f;
        error = clSetKernelArg(kernel, 3, sizeof(float), &gain);
    }
    if (!checkOpenCLError(error, "clSetKernelArg 3")) return false;
    
    // Execute kernel
    size_t globalWorkSize = numChannels * numSamples;
    size_t localWorkSize = 256;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    error = clEnqueueNDRangeKernel(m_queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, &m_event);
    if (!checkOpenCLError(error, "clEnqueueNDRangeKernel")) return false;
    
    clWaitForEvents(1, &m_event);
    
    auto end = std::chrono::high_resolution_clock::now();
    m_lastProcessingTime = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (m_event) {
        clReleaseEvent(m_event);
        m_event = nullptr;
    }
    
    return true;
}

} // namespace e47

#endif // AG_ENABLE_OPENCL
