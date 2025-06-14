# GPU Processing Support for AudioGridder

This document describes the GPU processing capabilities added to AudioGridder, enabling hardware-accelerated audio processing using CUDA and OpenCL.

## Overview

The GPU processing extension provides:
- **CUDA Support**: High-performance processing on NVIDIA GPUs
- **OpenCL Support**: Cross-platform GPU processing (NVIDIA, AMD, Intel)
- **Automatic Fallback**: Seamless fallback to CPU processing if GPU fails
- **Performance Monitoring**: Real-time GPU processing metrics
- **Memory Management**: Efficient GPU memory allocation and management

## Architecture

### Core Components

1. **GPUAudioProcessor** (Base Class)
   - Abstract interface for GPU audio processing
   - Device enumeration and management
   - Factory pattern for creating GPU processors

2. **CUDAAudioProcessor** (CUDA Implementation)
   - NVIDIA GPU-specific implementation
   - CUDA kernels for audio processing
   - cuBLAS integration for optimized operations

3. **OpenCLAudioProcessor** (OpenCL Implementation)
   - Cross-platform GPU implementation
   - OpenCL kernels for audio processing
   - Support for multiple GPU vendors

4. **AudioWorker Integration**
   - GPU processing integrated into audio pipeline
   - Automatic GPU/CPU fallback mechanism
   - Performance monitoring and logging

### Processing Pipeline

```
Audio Input → Channel Mapping → GPU Processing → Plugin Chain → Channel Unmapping → Audio Output
                                      ↓
                              CPU Fallback (if GPU fails)
```

## Build Configuration

### CMake Options

```bash
# Enable CUDA support
cmake .. -DAG_ENABLE_CUDA=ON

# Enable OpenCL support  
cmake .. -DAG_ENABLE_OPENCL=ON

# Enable both
cmake .. -DAG_ENABLE_CUDA=ON -DAG_ENABLE_OPENCL=ON
```

### Dependencies

**CUDA Support:**
- NVIDIA CUDA Toolkit 11.0+
- Compatible NVIDIA GPU with compute capability 3.0+

**OpenCL Support:**
- OpenCL 1.2+ runtime
- Compatible GPU (NVIDIA, AMD, Intel)

## Usage

### Initialization

```cpp
// Create AudioWorker instance
AudioWorker worker(&logTag);

// Initialize GPU processing (auto-detect best backend)
bool success = worker.initializeGPU(GPUBackend::AUTO);

// Or specify backend explicitly
bool success = worker.initializeGPU(GPUBackend::CUDA, 0); // Use CUDA device 0
```

### Device Enumeration

```cpp
// Get available GPU devices
auto devices = worker.getAvailableGPUDevices();

for (const auto& device : devices) {
    std::cout << "Device: " << device.name.toStdString() 
              << " (Backend: " << static_cast<int>(device.backend) << ")"
              << " Memory: " << device.totalMemory / (1024*1024) << " MB"
              << std::endl;
}
```

### Runtime Control

```cpp
// Enable/disable GPU processing
worker.setGPUProcessing(true);

// Check if GPU is enabled and working
if (worker.isGPUEnabled()) {
    std::cout << "GPU processing active" << std::endl;
}

// Shutdown GPU processing
worker.shutdownGPU();
```

## Performance Characteristics

### GPU Processing Benefits

1. **Parallel Processing**: Simultaneous processing of multiple audio samples
2. **High Throughput**: Optimized for large buffer sizes
3. **Low Latency**: Reduced CPU load allows for lower audio latencies
4. **Scalability**: Performance scales with GPU compute units

### Optimal Use Cases

- **Large Buffer Sizes**: 512+ samples per buffer
- **High Channel Counts**: 8+ audio channels
- **Complex Processing**: Multiple effects chains
- **High Sample Rates**: 96kHz+ audio processing

### Performance Monitoring

The system provides real-time performance metrics:

```cpp
// Get last processing time
double processingTime = gpuProcessor->getLastProcessingTimeMs();

// Get GPU memory usage
size_t memoryUsage = gpuProcessor->getMemoryUsage();
```

## Audio Processing Examples

### Basic Gain and Filtering

The current implementation includes example processing:

```cuda
// CUDA kernel example
__global__ void processAudioKernelFloat(float* data, int numChannels, int numSamples, float gain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSamples = numChannels * numSamples;
    
    if (idx < totalSamples) {
        // Apply gain
        data[idx] *= gain;
        
        // Simple high-pass filter
        if (idx > numChannels) {
            data[idx] = data[idx] - 0.95f * data[idx - numChannels];
        }
    }
}
```

### Extending Processing

To add custom GPU processing:

1. **Modify Kernels**: Update CUDA/OpenCL kernel code
2. **Add Parameters**: Extend kernel parameter passing
3. **Update Interface**: Add new processing methods to base class

## Error Handling

### Automatic Fallback

The system automatically falls back to CPU processing when:
- GPU initialization fails
- GPU processing encounters errors
- GPU memory allocation fails
- GPU device becomes unavailable

### Error Logging

All GPU operations include comprehensive error logging:

```cpp
// CUDA error checking
bool checkCudaError(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << operation << ": " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}
```

## Memory Management

### Buffer Allocation

GPU processors manage both device and host memory:

```cpp
// Allocate buffers for processing
bool allocateBuffers(int numChannels, int maxSamples) {
    // Device memory allocation
    cudaMalloc(&m_deviceBufferFloat, bufferSize);
    
    // Host pinned memory for fast transfers
    cudaMallocHost(&m_hostBufferFloat, bufferSize);
    
    return true;
}
```

### Memory Transfer Optimization

- **Pinned Memory**: Host memory is pinned for faster GPU transfers
- **Asynchronous Transfers**: Non-blocking memory operations
- **Interleaved Data**: Audio data is interleaved for optimal GPU access patterns

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - Verify GPU drivers are installed
   - Check CUDA/OpenCL runtime installation
   - Ensure GPU has sufficient compute capability

2. **Performance Issues**
   - Increase buffer size for better GPU utilization
   - Check GPU memory usage
   - Verify GPU is not thermal throttling

3. **Build Issues**
   - Ensure CUDA Toolkit is properly installed
   - Check OpenCL headers and libraries
   - Verify CMake can find GPU dependencies

### Debug Information

Enable verbose logging to diagnose issues:

```cpp
// GPU initialization will log detailed information
worker.initializeGPU(GPUBackend::AUTO);
```

## Future Enhancements

### Planned Features

1. **Advanced DSP Algorithms**
   - FFT-based processing
   - Convolution reverb
   - Spectral processing

2. **Multi-GPU Support**
   - Load balancing across multiple GPUs
   - GPU cluster processing

3. **Real-time Parameter Updates**
   - Dynamic kernel parameter updates
   - GPU-based automation

4. **Memory Pool Management**
   - Efficient memory reuse
   - Reduced allocation overhead

### Contributing

To contribute GPU processing improvements:

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include performance benchmarks
4. Update documentation
5. Test on multiple GPU vendors

## License

This GPU processing extension maintains the same MIT license as the main AudioGridder project.
