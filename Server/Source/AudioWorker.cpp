/*
 * Copyright (c) 2020 Andreas Pohl
 * Licensed under MIT (https://github.com/apohl79/audiogridder/blob/master/COPYING)
 *
 * Author: Andreas Pohl
 */

#include "AudioWorker.hpp"
#include <memory>
#include "Message.hpp"
#include "Defaults.hpp"
#include "App.hpp"
#include "Server.hpp"
#include "Metrics.hpp"
#include "Processor.hpp"

namespace e47 {

std::unordered_map<String, AudioWorker::RecentsListType> AudioWorker::m_recents;
std::mutex AudioWorker::m_recentsMtx;

AudioWorker::AudioWorker(LogTag* tag) : Thread("AudioWorker"), LogTagDelegate(tag), m_channelMapper(tag), 
    m_useGPUProcessing(false), m_gpuBackend(GPUBackend::NONE), m_gpuDeviceId(-1) {
    initAsyncFunctors();
}

AudioWorker::~AudioWorker() {
    traceScope();
    stopAsyncFunctors();
    shutdownGPU();
    if (nullptr != m_socket && m_socket->isConnected()) {
        m_socket->close();
    }
    waitForThreadAndLog(getLogTagSource(), this);
    m_socket.reset();
    m_chain.reset();
}

void AudioWorker::init(std::unique_ptr<StreamingSocket> s, HandshakeRequest cfg) {
    traceScope();
    m_socket = std::move(s);
    m_sampleRate = cfg.sampleRate;
    m_samplesPerBlock = cfg.samplesPerBlock;
    m_doublePrecision = cfg.doublePrecision;
    m_channelsIn = cfg.channelsIn;
    m_channelsOut = cfg.channelsOut;
    m_channelsSC = cfg.channelsSC;
    m_activeChannels = cfg.activeChannels;
    m_activeChannels.setWithInput(m_channelsIn > 0);
    m_activeChannels.setNumChannels(m_channelsIn + m_channelsSC, m_channelsOut);
    m_channelMapper.createServerMapping(m_activeChannels);
    m_channelMapper.print();
    m_chain = std::make_shared<ProcessorChain>(
        getLogTagSource(), ProcessorChain::createBussesProperties(m_channelsIn, m_channelsOut, m_channelsSC), cfg);
    if (m_doublePrecision && m_chain->supportsDoublePrecisionProcessing()) {
        m_chain->setProcessingPrecision(AudioProcessor::doublePrecision);
    }
    m_chain->updateChannels(m_channelsIn, m_channelsOut, m_channelsSC);
}

bool AudioWorker::waitForData() {
    std::lock_guard<std::mutex> lock(m_mtx);
    return m_socket->waitUntilReady(true, 50);
}

void AudioWorker::run() {
    traceScope();
    logln("audio processor started");

    AudioBuffer<float> bufferF;
    AudioBuffer<double> bufferD;
    MidiBuffer midi;
    AudioMessage msg(getLogTagSource());
    AudioPlayHead::PositionInfo posInfo;
    auto duration = TimeStatistic::getDuration("audio");
    auto bytesIn = Metrics::getStatistic<Meter>("NetBytesIn");
    auto bytesOut = Metrics::getStatistic<Meter>("NetBytesOut");

    ProcessorChain::PlayHead playHead(&posInfo);
    m_chain->prepareToPlay(m_sampleRate, m_samplesPerBlock);
    bool hasToSetPlayHead = true;

    auto traceCtx = TimeTrace::createTraceContext();
    Uuid traceId;

    auto processingThresholdMs = -1.0;
    if (auto srv = getApp()->getServer()) {
        processingThresholdMs = srv->getProcessingTraceTresholdMs();
    }
    if (processingThresholdMs <= 0.0) {
        processingThresholdMs = m_samplesPerBlock / m_sampleRate * 1000 - 1;
    }

    MessageHelper::Error e;
    while (!threadShouldExit() && isOk()) {
        // Read audio chunk
        if (waitForData()) {
            if (msg.readFromClient(m_socket.get(), bufferF, bufferD, midi, posInfo, &e, *bytesIn, traceId)) {
                traceCtx->reset(traceId);
                std::lock_guard<std::mutex> lock(m_mtx);
                traceCtx->add("aw_lock");
                duration.reset();
                if (hasToSetPlayHead) {  // do not set the playhead before it's initialized
                    m_chain->setPlayHead(&playHead);
                    hasToSetPlayHead = false;
                }
                int bufferChannels = msg.isDouble() ? bufferD.getNumChannels() : bufferF.getNumChannels();
                int neededChannels = m_activeChannels.getNumActiveChannels(true);
                if (neededChannels > bufferChannels) {
                    logln("error processing audio message: buffer has not enough channels: needed channels is "
                          << neededChannels << ", but buffer has " << bufferChannels);
                    m_chain->releaseResources();
                    m_socket->close();
                    break;
                }
                bool sendOk;
                if (msg.isDouble()) {
                    if (m_chain->supportsDoublePrecisionProcessing()) {
                        traceCtx->add("aw_prep");
                        traceCtx->startGroup();
                        processBlock(bufferD, midi);
                        traceCtx->finishGroup("aw_process");
                    } else {
                        bufferF.makeCopyOf(bufferD);
                        traceCtx->add("aw_prep");
                        traceCtx->startGroup();
                        processBlock(bufferF, midi);
                        traceCtx->finishGroup("aw_process");
                        bufferD.makeCopyOf(bufferF);
                    }
                    traceCtx->add("aw_finish");
                    sendOk = msg.sendToClient(m_socket.get(), bufferD, midi, m_chain->getLatencySamples(),
                                              bufferD.getNumChannels(), &e, *bytesOut);
                } else {
                    traceCtx->add("aw_prep");
                    traceCtx->startGroup();
                    processBlock(bufferF, midi);
                    traceCtx->finishGroup("aw_process");
                    sendOk = msg.sendToClient(m_socket.get(), bufferF, midi, m_chain->getLatencySamples(),
                                              bufferF.getNumChannels(), &e, *bytesOut);
                }
                traceCtx->summary(getLogTagSource(), "process audio", processingThresholdMs);
                if (!sendOk) {
                    logln("error: failed to send audio data to client: " << e.toString());
                    m_socket->close();
                }
                duration.update();
            } else {
                logln("error: failed to read audio message: " << e.toString());
                m_socket->close();
            }
        }
    }

    TimeTrace::deleteTraceContext();

    m_chain->setPlayHead(nullptr);

    duration.clear();
    clear();

    if (m_error.isNotEmpty()) {
        logln("audio processor error: " << m_error);
    }

    logln("audio processor terminated");
}

template <>
AudioBuffer<double>* AudioWorker::getProcBuffer() {
    return &m_procBufferD;
}

template <typename T>
void AudioWorker::processBlock(AudioBuffer<T>& buffer, MidiBuffer& midi) {
    int numChannels = jmax(m_channelsIn + m_channelsSC, m_channelsOut) + m_chain->getExtraChannels();
    
    // Try GPU processing first if enabled
    if (isGPUEnabled() && m_gpuProcessor->isInitialized()) {
        TimeTrace::addTracePoint("gpu_process_start");
        
        AudioBuffer<T>* targetBuffer = &buffer;
        AudioBuffer<T>* procBuffer = nullptr;
        
        if (numChannels > buffer.getNumChannels()) {
            // Need to use processing buffer for channel mapping
            procBuffer = getProcBuffer<T>();
            procBuffer->setSize(numChannels, buffer.getNumSamples());
            if (m_activeChannels.getNumActiveChannels(true) > 0) {
                m_channelMapper.map(&buffer, procBuffer);
                TimeTrace::addTracePoint("pb_ch_map");
            } else {
                procBuffer->clear();
            }
            targetBuffer = procBuffer;
        }
        
        // Process with GPU
        bool gpuSuccess = false;
        if constexpr (std::is_same_v<T, float>) {
            gpuSuccess = m_gpuProcessor->processAudioBuffer(static_cast<AudioBuffer<float>&>(*targetBuffer), targetBuffer->getNumSamples());
        } else if constexpr (std::is_same_v<T, double>) {
            gpuSuccess = m_gpuProcessor->processAudioBuffer(static_cast<AudioBuffer<double>&>(*targetBuffer), targetBuffer->getNumSamples());
        }
        
        if (gpuSuccess) {
            TimeTrace::addTracePoint("gpu_process_success");
            
            // Apply plugin chain processing after GPU processing
            m_chain->processBlock(*targetBuffer, midi);
            
            if (procBuffer) {
                m_channelMapper.mapReverse(procBuffer, &buffer);
                TimeTrace::addTracePoint("pb_ch_map_reverse");
            }
            
            TimeTrace::addTracePoint("gpu_process_complete");
            return;
        } else {
            logln("GPU processing failed, falling back to CPU");
            TimeTrace::addTracePoint("gpu_process_fallback");
        }
    }
    
    // CPU processing (original implementation)
    if (numChannels <= buffer.getNumChannels()) {
        m_chain->processBlock(buffer, midi);
    } else {
        // we received fewer channels, now we need to map the input/output data
        auto* procBuffer = getProcBuffer<T>();
        procBuffer->setSize(numChannels, buffer.getNumSamples());
        if (m_activeChannels.getNumActiveChannels(true) > 0) {
            m_channelMapper.map(&buffer, procBuffer);
            TimeTrace::addTracePoint("pb_ch_map");
        } else {
            procBuffer->clear();
        }
        m_chain->processBlock(*procBuffer, midi);
        m_channelMapper.mapReverse(procBuffer, &buffer);
        TimeTrace::addTracePoint("pb_ch_map_reverse");
    }
}

void AudioWorker::shutdown() {
    traceScope();
    signalThreadShouldExit();
}

void AudioWorker::clear() {
    traceScope();
    if (nullptr != m_chain) {
        m_chain->clear();
    }
}

bool AudioWorker::addPlugin(const String& id, const String& settings, const String& layout, uint64 monoChannels,
                            String& err) {
    traceScope();
    return m_chain->addPluginProcessor(id, settings, layout, monoChannels, err);
}

void AudioWorker::delPlugin(int idx) {
    traceScope();
    logln("deleting plugin " << idx);
    m_chain->delProcessor(idx);
}

void AudioWorker::exchangePlugins(int idxA, int idxB) {
    traceScope();
    logln("exchanging plugins idxA=" << idxA << " idxB=" << idxB);
    m_chain->exchangeProcessors(idxA, idxB);
}

String AudioWorker::getRecentsList(String host) const {
    traceScope();
    std::lock_guard<std::mutex> lock(m_recentsMtx);
    if (m_recents.find(host) == m_recents.end()) {
        return "";
    }
    auto& recents = m_recents[host];
    String list;
    for (auto& r : recents) {
        list += String(Processor::createJson(r).dump()) + "\n";
    }
    return list;
}

void AudioWorker::addToRecentsList(const String& id, const String& host) {
    traceScope();
    auto plug = Processor::findPluginDescription(id);
    if (plug != nullptr) {
        std::lock_guard<std::mutex> lock(m_recentsMtx);
        auto& recents = m_recents[host];
        recents.removeAllInstancesOf(*plug);
        recents.insert(0, *plug);
        int toRemove = recents.size() - Defaults::DEFAULT_NUM_RECENTS;
        if (toRemove > 0) {
            recents.removeLast(toRemove);
        }
    }
}

bool AudioWorker::initializeGPU(GPUBackend backend, int deviceId) {
    traceScope();
    
    if (m_gpuProcessor && m_gpuProcessor->isInitialized()) {
        logln("GPU already initialized");
        return true;
    }
    
    // Check if GPU is available
    if (!GPUAudioProcessor::isGPUAvailable()) {
        logln("No GPU devices available");
        return false;
    }
    
    // Create GPU processor
    m_gpuProcessor = GPUAudioProcessor::create(backend);
    if (!m_gpuProcessor) {
        logln("Failed to create GPU processor for backend: " << static_cast<int>(backend));
        return false;
    }
    
    // Initialize GPU processor
    if (!m_gpuProcessor->initialize(deviceId)) {
        logln("Failed to initialize GPU processor");
        m_gpuProcessor.reset();
        return false;
    }
    
    // Set processing precision
    m_gpuProcessor->setProcessingPrecision(m_doublePrecision);
    
    // Allocate GPU buffers
    int maxChannels = jmax(m_channelsIn + m_channelsSC, m_channelsOut);
    if (!m_gpuProcessor->allocateBuffers(maxChannels, m_samplesPerBlock)) {
        logln("Failed to allocate GPU buffers");
        m_gpuProcessor.reset();
        return false;
    }
    
    m_gpuBackend = backend;
    m_gpuDeviceId = deviceId;
    m_useGPUProcessing = true;
    
    logln("GPU processing initialized successfully - Backend: " << static_cast<int>(backend) 
          << ", Device: " << deviceId);
    
    return true;
}

void AudioWorker::shutdownGPU() {
    traceScope();
    
    if (m_gpuProcessor) {
        logln("Shutting down GPU processing");
        m_gpuProcessor->shutdown();
        m_gpuProcessor.reset();
    }
    
    m_useGPUProcessing = false;
    m_gpuBackend = GPUBackend::NONE;
    m_gpuDeviceId = -1;
}

std::vector<GPUDeviceInfo> AudioWorker::getAvailableGPUDevices() const {
    return GPUAudioProcessor::getAvailableDevices();
}

}  // namespace e47
