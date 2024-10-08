#include "DirectMLProcessor.hpp"

#include <algorithm>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <iostream>

#pragma warning(                                                                                                       \
    disable : 4238) // References to temporary classes are okay because they are only used as function parameters.

using Microsoft::WRL::ComPtr;

DirectMLProcessor::~DirectMLProcessor()
{
}

std::tuple<Microsoft::WRL::ComPtr<IDXCoreAdapter>, D3D_FEATURE_LEVEL> SelectAdapter(std::string_view adapterNameFilter)
{
    using Microsoft::WRL::ComPtr;

    ComPtr<IDXCoreAdapterFactory> adapterFactory;
    THROW_IF_FAILED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(adapterFactory.GetAddressOf())));

    // First try getting all GENERIC_ML devices, which is the broadest set of adapters
    // and includes both GPUs and NPUs; however, running this sample on an older build of
    // Windows may not have drivers that report GENERIC_ML.
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_1_0_GENERIC;
    ComPtr<IDXCoreAdapterList> adapterList;
    THROW_IF_FAILED(
        adapterFactory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, adapterList.GetAddressOf()));

    // Fall back to CORE_COMPUTE if GENERIC_ML devices are not available. This is a more restricted
    // set of adapters and may filter out some NPUs.
    if (adapterList->GetAdapterCount() == 0)
    {
        std::cout << "No GENERIC_ML adapters found. Falling back to CORE_COMPUTE.\n";
        featureLevel = D3D_FEATURE_LEVEL_1_0_CORE;
        THROW_IF_FAILED(adapterFactory->CreateAdapterList(1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE,
                                                          adapterList.GetAddressOf()));
    }

    if (adapterList->GetAdapterCount() == 0)
    {
        throw std::runtime_error("No compatible adapters found.");
    }

    // Sort the adapters by preference, with hardware and high-performance adapters first.
    DXCoreAdapterPreference preferences[] = {DXCoreAdapterPreference::Hardware,
                                             DXCoreAdapterPreference::HighPerformance};

    THROW_IF_FAILED(adapterList->Sort(_countof(preferences), preferences));

    std::vector<ComPtr<IDXCoreAdapter>> adapters;
    std::vector<std::string> adapterDescriptions;
    std::optional<uint32_t> firstAdapterMatchingNameFilter;

    for (uint32_t i = 0; i < adapterList->GetAdapterCount(); i++)
    {
        ComPtr<IDXCoreAdapter> adapter;
        THROW_IF_FAILED(adapterList->GetAdapter(i, adapter.GetAddressOf()));

        size_t descriptionSize;
        THROW_IF_FAILED(adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &descriptionSize));

        std::string adapterDescription(descriptionSize, '\0');
        THROW_IF_FAILED(
            adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, descriptionSize, adapterDescription.data()));

        // Remove trailing null terminator written by DXCore.
        while (!adapterDescription.empty() && adapterDescription.back() == '\0')
        {
            adapterDescription.pop_back();
        }

        if (adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE))
        {
            adapterDescription += " (CORE_COMPUTE)";
        }
        if (adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML))
        {
            adapterDescription += " (GENERIC_ML)";
        }

        adapters.push_back(adapter);
        adapterDescriptions.push_back(adapterDescription);

        if (!firstAdapterMatchingNameFilter && adapterDescription.find(adapterNameFilter) != std::string::npos)
        {
            firstAdapterMatchingNameFilter = i;
            std::cout << "Adapter[" << i << "]: " << adapterDescription << " (SELECTED)\n";
        }
        else
        {
            std::cout << "Adapter[" << i << "]: " << adapterDescription << "\n";
        }
    }

    if (!firstAdapterMatchingNameFilter)
    {
        throw std::invalid_argument("No adapters match the provided name filter.");
    }
    std::cout << "Selected adapter: " << adapterDescriptions[*firstAdapterMatchingNameFilter]
              << " index: " << *firstAdapterMatchingNameFilter << std::endl;
    return {adapters[*firstAdapterMatchingNameFilter], featureLevel};
}

void DirectMLProcessor::InitializeDirectML(std::string adapterNameFilter)
{
    auto [adapter, featureLevel] = SelectAdapter(adapterNameFilter);
    std::cout << "FeatureLevel: " << featureLevel << std::endl;
    Microsoft::WRL::ComPtr<ID3D12Device> d3d12Device;
    THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), featureLevel, IID_PPV_ARGS(&d3d12Device)));
    std::cout << "Direct3D 12 device created" << std::endl;
    Microsoft::WRL::ComPtr<IDMLDevice> dmlDevice;
    THROW_IF_FAILED(DMLCreateDevice(d3d12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmlDevice)));
    std::cout << "DirectML device created" << std::endl;

    m_d3D12Device = d3d12Device;
    m_dmlDevice = dmlDevice;

    // THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), featureLevel,
    //                                   IID_PPV_ARGS(m_d3D12Device.ReleaseAndGetAddressOf())));

    // THROW_IF_FAILED(
    //     DMLCreateDevice(m_d3D12Device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(m_dmlDevice.GetAddressOf())));

    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    THROW_IF_FAILED(
        m_d3D12Device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(m_commandQueue.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(m_d3D12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                          IID_PPV_ARGS(m_commandAllocator.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(m_d3D12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(),
                                                     nullptr, IID_PPV_ARGS(m_commandList.ReleaseAndGetAddressOf())));

    DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT fp16Query = {DML_TENSOR_DATA_TYPE_FLOAT16};
    DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT fp16Supported = {};
    THROW_IF_FAILED(m_dmlDevice->CheckFeatureSupport(DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(fp16Query),
                                                     &fp16Query, sizeof(fp16Supported), &fp16Supported));
    if (fp16Supported.IsSupported)
    {
        std::wcout << L"FP16 is supported." << std::endl;
    }
    else
    {
        std::wcout << L"FP16 is not supported." << std::endl;
    }

    DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT int8Query = {DML_TENSOR_DATA_TYPE_INT8};
    DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT int8Supported = {};
    THROW_IF_FAILED(m_dmlDevice->CheckFeatureSupport(DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(int8Query),
                                                     &int8Query, sizeof(int8Supported), &int8Supported));
    if (int8Supported.IsSupported)
    {
        std::wcout << L"INT8 is supported." << std::endl;
    }
    else
    {
        std::wcout << L"INT8 is not supported." << std::endl;
    }
    // for (int i = DML_TENSOR_DATA_TYPE_UNKNOWN; i <= DML_TENSOR_DATA_TYPE_INT64; ++i)
    // {
    //     DML_TENSOR_DATA_TYPE type = static_cast<DML_TENSOR_DATA_TYPE>(i);
    //     std::cout << "Enum value: " << type << std::endl;
    // }
}

void DirectMLProcessor::CloseExecuteResetWait()
{
    THROW_IF_FAILED(m_commandList->Close());

    ID3D12CommandList *commandLists[] = {m_commandList.Get()};
    m_commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

    ComPtr<ID3D12Fence> d3D12Fence;
    THROW_IF_FAILED(m_d3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(d3D12Fence.GetAddressOf())));

    wil::unique_handle fenceEventHandle(::CreateEvent(nullptr, true, false, nullptr));
    THROW_LAST_ERROR_IF_NULL(fenceEventHandle);

    THROW_IF_FAILED(m_commandQueue->Signal(d3D12Fence.Get(), 1));
    THROW_IF_FAILED(d3D12Fence->SetEventOnCompletion(1, fenceEventHandle.get()));

    ::WaitForSingleObjectEx(fenceEventHandle.get(), INFINITE, FALSE);

    THROW_IF_FAILED(m_commandAllocator->Reset());
    THROW_IF_FAILED(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
}

void DirectMLProcessor::SetTensorData(std::string name, uint32_t *shapes, DML_TENSOR_DATA_TYPE type, const void *data,
                                      size_t size)
{
    std::cout << "Enter SetTensorData " << name << std::endl;

    ComPtr<ID3D12Resource> tensorResource, uploadBuffer;
    if (m_tensorInfoMap.find(name) == m_tensorInfoMap.end())
    {
        std::cout << "Tensor " << name << " not found" << std::endl;
        m_tensorInfoMap[name] = new TensorInfo();
        m_tensorInfoMap[name]->dimensions = {shapes[0], shapes[1], shapes[2], shapes[3]};
        m_tensorInfoMap[name]->elementCount = shapes[0] * shapes[1] * shapes[2] * shapes[3];
        m_tensorInfoMap[name]->desc = {type, m_tensorInfoMap[name]->dimensions};
        std::wcout << "Tensor Buffer Size: " << m_tensorInfoMap[name]->desc.totalTensorSizeInBytes << std::endl;

        THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(m_tensorInfoMap[name]->desc.totalTensorSizeInBytes,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(tensorResource.GetAddressOf())));
        m_tensorInfoMap[name]->resource = tensorResource;

        THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(m_tensorInfoMap[name]->desc.totalTensorSizeInBytes),
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(uploadBuffer.GetAddressOf())));
        m_tensorInfoMap[name]->uploadResource = uploadBuffer;
    }
    else
    {
        std::cout << "Tensor " << name << " already exists" << std::endl;
        tensorResource = m_tensorInfoMap[name]->resource;
        uploadBuffer = m_tensorInfoMap[name]->uploadResource;
    }

    D3D12_SUBRESOURCE_DATA tensorSubresourceData{};
    tensorSubresourceData.pData = data;
    tensorSubresourceData.RowPitch = static_cast<LONG_PTR>(size);
    tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

    // Upload the input tensor to the GPU.
    ::UpdateSubresources(m_commandList.Get(), tensorResource.Get(), uploadBuffer.Get(), 0, 0, 1,
                         &tensorSubresourceData);

    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(tensorResource.Get(),
                                                                            D3D12_RESOURCE_STATE_COPY_DEST,
                                                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
}

void DirectMLProcessor::GetTensorData(std::string name, uint32_t *shapes, DML_TENSOR_DATA_TYPE type, void *data,
                                      size_t size)
{
    std::cout << "Enter GetTensorData " << name << std::endl;

    if (m_tensorInfoMap.find(name) == m_tensorInfoMap.end())
    {
        std::cout << "Tensor " << name << " not found" << std::endl;
        return;
    }
    // The output buffer now contains the result of the identity operator,
    // so read it back if you want the CPU to access it.
    ComPtr<ID3D12Resource> readbackBuffer;
    THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK), D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(m_tensorInfoMap[name]->desc.totalTensorSizeInBytes),
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(readbackBuffer.GetAddressOf())));

    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_tensorInfoMap[name]->resource.Get(),
                                                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                                            D3D12_RESOURCE_STATE_COPY_SOURCE));

    m_commandList->CopyResource(readbackBuffer.Get(), m_tensorInfoMap[name]->resource.Get());

    CloseExecuteResetWait();

    D3D12_RANGE tensorBufferRange{0, static_cast<SIZE_T>(m_tensorInfoMap[name]->desc.totalTensorSizeInBytes)};
    FLOAT *outputBufferData{};
    THROW_IF_FAILED(readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void **>(&outputBufferData)));

    memcpy(data, outputBufferData, size);

    D3D12_RANGE emptyRange{0, 0};
    readbackBuffer->Unmap(0, &emptyRange);
}

bool DirectMLProcessor::CanBroadcast(const dml::TensorDimensions &a, const dml::TensorDimensions &b)
{
    if (a != b)
    {
        return (a[0] % b[0] == 0) && (a[1] % b[1] == 0) && (a[2] % b[2] == 0) && (a[3] % b[3] == 0);
    }
    else
    {
        return true;
    }
}

dml::Expression DirectMLProcessor::ReshapeAndBroadcastTensor(dml::Expression originalTensor,
                                                             const dml::TensorDimensions &targetShape)
{

    const auto &originalShape = originalTensor.GetOutputDesc().sizes;

    // Step 1: Reshape the original tensor if necessary
    dml::Expression reshapedTensor = originalTensor;
    if (originalShape.size() != targetShape.size())
    {
        // Pad the original shape with 1s to match the target shape's rank
        dml::TensorDimensions paddedShape(targetShape.size(), 1);
        std::copy(originalShape.rbegin(), originalShape.rend(), paddedShape.rbegin());
        reshapedTensor = dml::Reinterpret(originalTensor, paddedShape, dml::NullOpt);
    }

    // Step 2: Broadcast the reshaped tensor
    dml::TensorDimensions repeats(targetShape.size(), 1);
    for (size_t i = 0; i < targetShape.size(); ++i)
    {
        if (reshapedTensor.GetOutputDesc().sizes[i] == 1 && targetShape[i] != 1)
        {
            repeats[i] = targetShape[i];
        }
    }

    return dml::Tile(reshapedTensor, repeats);
}

void DirectMLProcessor::ElementWiseAddBcast(std::string src0, std::string src1, std::string dst)
{
    if (m_tensorInfoMap.find(src0) == m_tensorInfoMap.end() || m_tensorInfoMap.find(src1) == m_tensorInfoMap.end() ||
        m_tensorInfoMap.find(dst) == m_tensorInfoMap.end())
    {
        std::cout << "Tensor not found" << std::endl;
        return;
    }

    dml::Graph graph(m_dmlDevice.Get());
    dml::Expression input = dml::InputTensor(graph, 0, m_tensorInfoMap[src0]->desc);
    dml::Expression input2;
    if (m_tensorInfoMap[src0]->dimensions != m_tensorInfoMap[src1]->dimensions)
    {
        assert(CanBroadcast(m_tensorInfoMap[src0]->dimensions, m_tensorInfoMap[src1]->dimensions));
        std::cout << "Tensor shapes do not match, might need Broadcasting" << std::endl;
        input2 = ReshapeAndBroadcastTensor(dml::InputTensor(graph, 1, m_tensorInfoMap[src1]->desc),
                                           m_tensorInfoMap[src0]->dimensions);
    }
    else
    {
        input2 = dml::InputTensor(graph, 1, m_tensorInfoMap[src1]->desc);
    }
    dml::Expression output = dml::Add(input, input2);

    ComPtr<IDMLCompiledOperator> dmlCompiledOperator;
    DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    dmlCompiledOperator.Attach(graph.Compile(executionFlags, {output}).Detach());

    ComPtr<IDMLOperatorInitializer> dmlOpInitializer;
    IDMLCompiledOperator *dmlCompiledOperators[] = {dmlCompiledOperator.Get()};
    THROW_IF_FAILED(m_dmlDevice->CreateOperatorInitializer(ARRAYSIZE(dmlCompiledOperators), dmlCompiledOperators,
                                                           IID_PPV_ARGS(dmlOpInitializer.GetAddressOf())));

    DML_BINDING_PROPERTIES initializeBindingProperties = dmlOpInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProperties = dmlCompiledOperator->GetBindingProperties();
    UINT descriptorCount =
        std::max(initializeBindingProperties.RequiredDescriptorCount, executeBindingProperties.RequiredDescriptorCount);

    // Create descriptor heaps.
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(
        m_d3D12Device->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(descriptorHeap.GetAddressOf())));

    // Set the descriptor heap(s).
    ID3D12DescriptorHeap *d3D12DescriptorHeaps[] = {descriptorHeap.Get()};
    m_commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Create a binding table over the descriptor heap we just created.
    DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
    dmlBindingTableDesc.Dispatchable = dmlOpInitializer.Get();
    dmlBindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.SizeInDescriptors = descriptorCount;

    ComPtr<IDMLBindingTable> dmlBindingTable;
    THROW_IF_FAILED(
        m_dmlDevice->CreateBindingTable(&dmlBindingTableDesc, IID_PPV_ARGS(dmlBindingTable.GetAddressOf())));

    UINT64 temporaryResourceSize =
        std::max(initializeBindingProperties.TemporaryResourceSize, executeBindingProperties.TemporaryResourceSize);

    ComPtr<ID3D12Resource> temporaryBuffer;
    if (temporaryResourceSize != 0)
    {
        THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(temporaryBuffer.GetAddressOf())));

        if (initializeBindingProperties.TemporaryResourceSize != 0)
        {
            DML_BUFFER_BINDING bufferBinding{temporaryBuffer.Get(), 0, temporaryResourceSize};
            DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
            dmlBindingTable->BindTemporaryResource(&bindingDesc);
        }
    }

    UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;
    std::wcout << L"presistentResourceSize: " << L' ' << persistentResourceSize << std::endl;
    ComPtr<ID3D12Resource> persistentBuffer;
    if (persistentResourceSize != 0)
    {
        THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(persistentResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(persistentBuffer.GetAddressOf())));

        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING bufferBinding{persistentBuffer.Get(), 0, persistentResourceSize};
        DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
        dmlBindingTable->BindOutputs(1, &bindingDesc);
    }

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    ComPtr<IDMLCommandRecorder> dmlCommandRecorder;
    THROW_IF_FAILED(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(dmlCommandRecorder.GetAddressOf())));

    // Record execution of the operator initializer.
    dmlCommandRecorder->RecordDispatch(m_commandList.Get(), dmlOpInitializer.Get(), dmlBindingTable.Get());

    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could
    // in principle record the execution into the same command list as the initialization, but you need only to
    // Initialize once, and typically you want to Execute an operator more frequently than that.
    CloseExecuteResetWait();

    //
    // Bind and execute the operator on the GPU.
    //

    m_commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Reset the binding table to bind for the operator we want to execute (it was previously used to bind for the
    // initializer).

    dmlBindingTableDesc.Dispatchable = dmlCompiledOperator.Get();

    THROW_IF_FAILED(dmlBindingTable->Reset(&dmlBindingTableDesc));

    if (temporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{temporaryBuffer.Get(), 0, temporaryResourceSize};
        DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
        dmlBindingTable->BindTemporaryResource(&bindingDesc);
    }

    if (persistentResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{persistentBuffer.Get(), 0, persistentResourceSize};
        DML_BINDING_DESC bindingDesc{DML_BINDING_TYPE_BUFFER, &bufferBinding};
        dmlBindingTable->BindPersistentResource(&bindingDesc);
    }

    // Create tensor buffers for output/readback of the tensor elements.

    DML_BUFFER_BINDING inputBufferBinding{m_tensorInfoMap[src0]->resource.Get(), 0,
                                          m_tensorInfoMap[src0]->desc.totalTensorSizeInBytes};
    DML_BINDING_DESC inputBindingDesc{DML_BINDING_TYPE_BUFFER, &inputBufferBinding};

    DML_BUFFER_BINDING inputBufferBinding2{m_tensorInfoMap[src1]->resource.Get(), 0,
                                           m_tensorInfoMap[src1]->desc.totalTensorSizeInBytes};
    DML_BINDING_DESC inputBindingDesc2{DML_BINDING_TYPE_BUFFER, &inputBufferBinding2};

    DML_BINDING_DESC inputBindingDescs[] = {inputBindingDesc, inputBindingDesc2};
    dmlBindingTable->BindInputs(2, inputBindingDescs);
    std::cout << "Binding inputs" << std::endl;

    DML_BUFFER_BINDING outputBufferBinding{m_tensorInfoMap[dst]->resource.Get(), 0,
                                           m_tensorInfoMap[dst]->desc.totalTensorSizeInBytes};
    DML_BINDING_DESC outputBindingDesc{DML_BINDING_TYPE_BUFFER, &outputBufferBinding};
    dmlBindingTable->BindOutputs(1, &outputBindingDesc);
    std::cout << "Binding outputs" << std::endl;

    // Record execution of the compiled operator.
    dmlCommandRecorder->RecordDispatch(m_commandList.Get(), dmlCompiledOperator.Get(), dmlBindingTable.Get());
    std::cout << "Record Dispatch" << std::endl;

    CloseExecuteResetWait();
    std::wcout << "Exit ElementWiseAddBcast" << std::endl;
}

void DirectMLProcessor::FreeResources()
{
    for (auto &tensor : m_tensorInfoMap)
    {
        tensor.second->resource.Reset();
        tensor.second->uploadResource.Reset();
        delete tensor.second;
    }
    m_tensorInfoMap.clear();
}
