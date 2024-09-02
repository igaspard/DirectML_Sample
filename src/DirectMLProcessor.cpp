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

void DirectMLProcessor::InitializeDirectML(bool forceNpu)
{
    ComPtr<IDXCoreAdapterFactory> factory;
    ::DXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory));
    ComPtr<IDXCoreAdapter> adapter;
    if (factory)
    {
        const GUID dxGUIDs[] = {DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE};
        ComPtr<IDXCoreAdapterList> adapterList;
        THROW_IF_FAILED(factory->CreateAdapterList(ARRAYSIZE(dxGUIDs), dxGUIDs, IID_PPV_ARGS(&adapterList)));
        for (uint32_t i = 0, adapterCount = adapterList->GetAdapterCount(); i < adapterCount; i++)
        {
            ComPtr<IDXCoreAdapter> currentGpuAdapter;
            THROW_IF_FAILED(adapterList->GetAdapter(static_cast<uint32_t>(i), IID_PPV_ARGS(&currentGpuAdapter)));

            if (forceNpu && currentGpuAdapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE))
            {
                adapter = std::move(currentGpuAdapter);
                break;
            }
            else
            {
                adapter = std::move(currentGpuAdapter);
                break;
            }
        }
    }

    THROW_IF_FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_1_0_CORE,
                                      IID_PPV_ARGS(m_d3D12Device.ReleaseAndGetAddressOf())));

    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    THROW_IF_FAILED(
        m_d3D12Device->CreateCommandQueue(&commandQueueDesc, IID_PPV_ARGS(m_commandQueue.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(m_d3D12Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                          IID_PPV_ARGS(m_commandAllocator.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(m_d3D12Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_commandAllocator.Get(),
                                                     nullptr, IID_PPV_ARGS(m_commandList.ReleaseAndGetAddressOf())));

    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
#if defined(_DEBUG)
    // If the project is in a debug build, then enable debugging via DirectML debug layers with this flag.
    dmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif
    THROW_IF_FAILED(
        DMLCreateDevice(m_d3D12Device.Get(), dmlCreateDeviceFlags, IID_PPV_ARGS(m_dmlDevice.GetAddressOf())));

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

        for (int i = 0; i < 4; i++)
            m_tensorInfoMap[name]->shapes[i] = shapes[i];

        m_tensorInfoMap[name]->elementCount = shapes[0] * shapes[1] * shapes[2] * shapes[3];
        m_tensorInfoMap[name]->desc = {DML_TENSOR_DATA_TYPE_FLOAT32, {shapes[0], shapes[1], shapes[2], shapes[3]}};
        std::wcout << "Tensor Buffer Size: " << m_tensorInfoMap[name]->desc.totalTensorSizeInBytes << std::endl;

        THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT), D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(m_tensorInfoMap[name]->desc.totalTensorSizeInBytes,
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(tensorResource.GetAddressOf())));
        m_tensorInfoMap[name]->resource = tensorResource;
        std::cout << "Tensor " << name << " created: " << m_tensorInfoMap[name]->resource.GetAddressOf() << std::endl;

        THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(m_tensorInfoMap[name]->desc.totalTensorSizeInBytes),
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(uploadBuffer.GetAddressOf())));
        m_tensorInfoMap[name]->uploadResource = uploadBuffer;
        std::cout << "Upload Buffer created: " << m_tensorInfoMap[name]->uploadResource.GetAddressOf() << std::endl;
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
        &CD3DX12_RESOURCE_DESC::Buffer(m_tensorInfoMap[name]->desc.totalTensorSizeInBytes), D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(readbackBuffer.GetAddressOf())));

    m_commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(m_tensorInfoMap[name]->resource.Get(),
                                                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                                            D3D12_RESOURCE_STATE_COPY_SOURCE));

    m_commandList->CopyResource(readbackBuffer.Get(), m_tensorInfoMap[name]->resource.Get());

    CloseExecuteResetWait();

    D3D12_RANGE tensorBufferRange{0, static_cast<SIZE_T>(m_tensorInfoMap[name]->desc.totalTensorSizeInBytes)};
    FLOAT *outputBufferData{};
    THROW_IF_FAILED(readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void **>(&outputBufferData)));

    memcpy(data, outputBufferData, size);

    // std::wstring outputString = L"output tensor: ";
    // for (size_t tensorElementIndex{0}; tensorElementIndex < m_tensorInfoMap[name]->elementCount;
    //      ++tensorElementIndex, ++outputBufferData)
    // {
    //     outputString += std::to_wstring(*outputBufferData) + L' ';
    // }

    // std::wcout << outputString << std::endl;
    // OutputDebugStringW(outputString.c_str());

    D3D12_RANGE emptyRange{0, 0};
    readbackBuffer->Unmap(0, &emptyRange);
}

void DirectMLProcessor::ElementWiseAdd(std::string src0, std::string src1, std::string dst)
{
    if (m_tensorInfoMap.find(src0) == m_tensorInfoMap.end() || m_tensorInfoMap.find(src1) == m_tensorInfoMap.end() ||
        m_tensorInfoMap.find(dst) == m_tensorInfoMap.end()) 
    {
        std::cout << "Tensor not found" << std::endl;
        return;
    }
    
    dml::Graph graph(m_dmlDevice.Get());
    dml::Expression input = dml::InputTensor(graph, 0, m_tensorInfoMap[src0]->desc);
    dml::Expression input2 = dml::InputTensor(graph, 1, m_tensorInfoMap[src1]->desc);
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
    
    DML_BUFFER_BINDING inputBufferBinding{m_tensorInfoMap[src0]->resource.Get(), 0, m_tensorInfoMap[src0]->desc.totalTensorSizeInBytes};
    DML_BINDING_DESC inputBindingDesc{DML_BINDING_TYPE_BUFFER, &inputBufferBinding};

    DML_BUFFER_BINDING inputBufferBinding2{m_tensorInfoMap[src1]->resource.Get(), 0, m_tensorInfoMap[src1]->desc.totalTensorSizeInBytes};
    DML_BINDING_DESC inputBindingDesc2{DML_BINDING_TYPE_BUFFER, &inputBufferBinding2};

    DML_BINDING_DESC inputBindingDescs[] = {inputBindingDesc, inputBindingDesc2};
    dmlBindingTable->BindInputs(2, inputBindingDescs);
    std::cout << "Binding inputs" << std::endl;

    DML_BUFFER_BINDING outputBufferBinding{m_tensorInfoMap[dst]->resource.Get(), 0, m_tensorInfoMap[dst]->desc.totalTensorSizeInBytes};
    DML_BINDING_DESC outputBindingDesc{DML_BINDING_TYPE_BUFFER, &outputBufferBinding};
    dmlBindingTable->BindOutputs(1, &outputBindingDesc);
    std::cout << "Binding outputs" << std::endl;

    // Record execution of the compiled operator.
    dmlCommandRecorder->RecordDispatch(m_commandList.Get(), dmlCompiledOperator.Get(), dmlBindingTable.Get());
    std::cout << "Record Dispatch" << std::endl;

    CloseExecuteResetWait();
    std::wcout << "Exit ElementWiseAdd" << std::endl;
}
