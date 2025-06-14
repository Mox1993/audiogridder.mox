/*
 * GPU Processing Test for AudioGridder
 * 
 * This test program demonstrates the GPU processing capabilities
 * and can be used to verify the implementation works correctly.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <memory>

// Mock JUCE types for testing (in real implementation, these come from JUCE)
#ifndef JUCE_HEADER_INCLUDED
class String {
public:
    String() = default;
    String(const char* str) : data(str) {}
    String(const std::string& str) : data(str) {}
    
    std::string toStdString() const { return data; }
    const char* toRawUTF8() const { return data.c_str(); }
    
private:
    std::string data;
};

template<typename T>
class AudioBuffer {
public:
    AudioBuffer(int channels, int samples) : m_channels(channels), m_samples(samples) {
        m_data.resize(channels);
        for (auto& channel : m_data) {
            channel.resize(samples, 0.0f);
        }
    }
    
    int getNumChannels() const { return m_channels; }
    int getNumSamples() const { return m_samples; }
    
    const T* getReadPointer(int channel) const { return m_data[channel].data(); }
    T* getWritePointer(int channel) { return m_data[channel].data(); }
    
    void setSize(int channels, int samples) {
        m_channels = channels;
        m_samples = samples;
        m_data.resize(channels);
        for (auto& channel : m_data) {
            channel.resize(samples);
        }
    }
    
private:
    int m_channels, m_samples;
    std::vector<std::vector<T>> m_data;
};
#endif

// Include GPU processor headers
#include "GPUAudioProcessor.hpp"

using namespace e47;

class GPUProcessingTest {
public:
    void runTests() {
        std::cout << "=== AudioGridder GPU Processing Test ===" << std::endl;
        
        testDeviceEnumeration();
        testGPUProcessorCreation();
        testAudioProcessing();
        testPerformanceBenchmark();
        
        std::cout << "=== All tests completed ===" << std::endl;
    }
    
private:
    void testDeviceEnumeration() {
        std::cout << "\n1. Testing GPU Device Enumeration..." << std::endl;
        
        auto devices = GPUAudioProcessor::getAvailableDevices();
        
        std::cout << "Found " << devices.size() << " GPU device(s):" << std::endl;
        
        for (size_t i = 0; i < devices.size(); ++i) {
            const auto& device = devices[i];
            std::cout << "  Device " << i << ":" << std::endl;
            std::cout << "    Name: " << device.name.toStdString() << std::endl;
            std::cout << "    Backend: " << getBackendName(device.backend) << std::endl;
            std::cout << "    Total Memory: " << (device.totalMemory / (1024*1024)) << " MB" << std::endl;
            std::cout << "    Free Memory: " << (device.freeMemory / (1024*1024)) << " MB" << std::endl;
            std::cout << "    Compute Capability: " << device.computeCapability << std::endl;
            std::cout << "    Available: " << (device.isAvailable ? "Yes" : "No") << std::endl;
        }
        
        if (devices.empty()) {
            std::cout << "  No GPU devices found. GPU processing will not be available." << std::endl;
        }
    }
    
    void testGPUProcessorCreation() {
        std::cout << "\n2. Testing GPU Processor Creation..." << std::endl;
        
        if (!GPUAudioProcessor::isGPUAvailable()) {
            std::cout << "  GPU not available, skipping processor creation test." << std::endl;
            return;
        }
        
        // Test auto backend selection
        auto processor = GPUAudioProcessor::create(GPUBackend::AUTO);
        if (processor) {
            std::cout << "  Successfully created GPU processor with AUTO backend" << std::endl;
            
            if (processor->initialize()) {
                std::cout << "  GPU processor initialized successfully" << std::endl;
                
                // Test buffer allocation
                if (processor->allocateBuffers(2, 512)) {
                    std::cout << "  GPU buffers allocated successfully (2 channels, 512 samples)" << std::endl;
                    std::cout << "  Memory usage: " << (processor->getMemoryUsage() / 1024) << " KB" << std::endl;
                } else {
                    std::cout << "  Failed to allocate GPU buffers" << std::endl;
                }
                
                processor->shutdown();
            } else {
                std::cout << "  Failed to initialize GPU processor" << std::endl;
            }
        } else {
            std::cout << "  Failed to create GPU processor" << std::endl;
        }
    }
    
    void testAudioProcessing() {
        std::cout << "\n3. Testing Audio Processing..." << std::endl;
        
        if (!GPUAudioProcessor::isGPUAvailable()) {
            std::cout << "  GPU not available, skipping audio processing test." << std::endl;
            return;
        }
        
        auto processor = GPUAudioProcessor::create(GPUBackend::AUTO);
        if (!processor || !processor->initialize()) {
            std::cout << "  Failed to create/initialize GPU processor" << std::endl;
            return;
        }
        
        const int channels = 2;
        const int samples = 512;
        
        if (!processor->allocateBuffers(channels, samples)) {
            std::cout << "  Failed to allocate GPU buffers" << std::endl;
            return;
        }
        
        // Create test audio buffer with sine wave
        AudioBuffer<float> buffer(channels, samples);
        fillWithSineWave(buffer);
        
        std::cout << "  Processing " << channels << " channels, " << samples << " samples..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        bool success = processor->processAudioBuffer(buffer, samples);
        auto end = std::chrono::high_resolution_clock::now();
        
        if (success) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "  GPU processing successful!" << std::endl;
            std::cout << "  Processing time: " << duration.count() << " microseconds" << std::endl;
            std::cout << "  GPU reported time: " << processor->getLastProcessingTimeMs() << " ms" << std::endl;
        } else {
            std::cout << "  GPU processing failed" << std::endl;
        }
        
        processor->shutdown();
    }
    
    void testPerformanceBenchmark() {
        std::cout << "\n4. Performance Benchmark..." << std::endl;
        
        if (!GPUAudioProcessor::isGPUAvailable()) {
            std::cout << "  GPU not available, skipping benchmark." << std::endl;
            return;
        }
        
        auto processor = GPUAudioProcessor::create(GPUBackend::AUTO);
        if (!processor || !processor->initialize()) {
            std::cout << "  Failed to create/initialize GPU processor" << std::endl;
            return;
        }
        
        const int channels = 8;
        const int samples = 1024;
        const int iterations = 100;
        
        if (!processor->allocateBuffers(channels, samples)) {
            std::cout << "  Failed to allocate GPU buffers" << std::endl;
            return;
        }
        
        AudioBuffer<float> buffer(channels, samples);
        fillWithSineWave(buffer);
        
        std::cout << "  Running benchmark: " << iterations << " iterations, " 
                  << channels << " channels, " << samples << " samples" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            processor->processAudioBuffer(buffer, samples);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avgTime = static_cast<double>(totalTime.count()) / iterations;
        double samplesPerSecond = (channels * samples * 1000000.0) / avgTime;
        
        std::cout << "  Average processing time: " << avgTime << " microseconds" << std::endl;
        std::cout << "  Throughput: " << (samplesPerSecond / 1000000.0) << " million samples/second" << std::endl;
        
        processor->shutdown();
    }
    
    void fillWithSineWave(AudioBuffer<float>& buffer) {
        const float frequency = 440.0f; // A4
        const float sampleRate = 44100.0f;
        const float amplitude = 0.5f;
        
        for (int channel = 0; channel < buffer.getNumChannels(); ++channel) {
            float* channelData = buffer.getWritePointer(channel);
            for (int sample = 0; sample < buffer.getNumSamples(); ++sample) {
                float phase = 2.0f * M_PI * frequency * sample / sampleRate;
                channelData[sample] = amplitude * std::sin(phase);
            }
        }
    }
    
    std::string getBackendName(GPUBackend backend) {
        switch (backend) {
            case GPUBackend::CUDA: return "CUDA";
            case GPUBackend::OPENCL: return "OpenCL";
            case GPUBackend::AUTO: return "Auto";
            case GPUBackend::NONE: return "None";
            default: return "Unknown";
        }
    }
};

int main() {
    try {
        GPUProcessingTest test;
        test.runTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
