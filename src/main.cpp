#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

// Function to set up the Metal device and command queue
void setupMetal(MTL::Device*& device, MTL::CommandQueue*& commandQueue) {
    device = MTL::CreateSystemDefaultDevice();
    commandQueue = device->newCommandQueue();
}

// Function to create and compile the compute shader
MTL::Function* compileShader(MTL::Device* device, const char* shaderSource) {
    NS::Error* error = nullptr;
    MTL::Library* library = device->newLibrary(NS::String::string(shaderSource, NS::UTF8StringEncoding), nullptr, &error);
    if (!library) {
        std::cerr << "Failed to create library: " << error->localizedDescription()->utf8String() << std::endl;
        return nullptr;
    }

    return library->newFunction(NS::String::string("compute_function", NS::UTF8StringEncoding));
}

std::string readMSLFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main(int argc, char* argv[]) {
    std::filesystem::path shaderRootDir;
    // Check if a shader root directory was provided as a command-line argument
    if (argc > 1) {
        shaderRootDir = argv[1];
    } else {
        std::cout<<"Provide shader root directory!";
        return 1;
    }

    MTL::Device* device = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    setupMetal(device, commandQueue);

    // Define the compute shader
    std::string shaderSource;
    try {
        shaderSource = readMSLFile(shaderRootDir.string() + "/compute.msl");
    } catch (const std::exception& e) {
        std::cerr << "Error reading shader file: " << e.what() << std::endl;
        return 1;
    }

    // Compile the shader
    MTL::Function* computeFunction = compileShader(device, shaderSource.c_str());
    if (!computeFunction) {
        return 1;
    }

    // Create the compute pipeline state
    NS::Error* error = nullptr;
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(computeFunction, &error);
    if (!pipelineState) {
        std::cerr << "Failed to create pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    // Prepare input data
    const int dataSize = 1000;
    std::vector<float> inputData(dataSize);
    for (int i = 0; i < dataSize; ++i) {
        inputData[i] = static_cast<float>(i);
    }

    // Create input and output buffers
    MTL::Buffer* inputBuffer = device->newBuffer(inputData.data(), dataSize * sizeof(float), MTL::ResourceStorageModeShared);
    MTL::Buffer* outputBuffer = device->newBuffer(dataSize * sizeof(float), MTL::ResourceStorageModeShared);

    // Create a command buffer and encoder
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();

    // Set the compute pipeline state
    computeEncoder->setComputePipelineState(pipelineState);

    // Set buffers
    computeEncoder->setBuffer(inputBuffer, 0, 0);
    computeEncoder->setBuffer(outputBuffer, 0, 1);

    // Dispatch threads
    MTL::Size gridSize = MTL::Size(dataSize, 1, 1);
    MTL::Size threadgroupSize = MTL::Size(pipelineState->maxTotalThreadsPerThreadgroup(), 1, 1);
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);

    // End encoding and commit the command buffer
    computeEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // Read the results
    float* outputData = static_cast<float*>(outputBuffer->contents());
    for (int i = 0; i < 10; ++i) {
        std::cout << "Output[" << i << "] = " << outputData[i] << std::endl;
    }

    // Clean up
    pipelineState->release();
    inputBuffer->release();
    outputBuffer->release();
    computeFunction->release();
    commandQueue->release();
    device->release();

    return 0;
}