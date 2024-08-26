#include "DirectMLProcessor.hpp"

#include <dxcore_interface.h>
#include <dxcore.h>
#include <iostream>
#include <algorithm>

#pragma warning(disable : 4238) // References to temporary classes are okay because they are only used as function parameters.

using Microsoft::WRL::ComPtr;


DirectMLProcessor::DirectMLProcessor(bool forceNpu)
{
    InitializeDirectML(forceNpu);
}

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
        const GUID dxGUIDs[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
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
    
    THROW_IF_FAILED(D3D12CreateDevice(
        adapter.Get(),
        D3D_FEATURE_LEVEL_1_0_CORE,
        IID_PPV_ARGS(m_d3D12Device.ReleaseAndGetAddressOf())));

    D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
    commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

    THROW_IF_FAILED(m_d3D12Device->CreateCommandQueue(
        &commandQueueDesc,
        IID_PPV_ARGS(m_commandQueue.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(m_d3D12Device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(m_commandAllocator.ReleaseAndGetAddressOf())));

    THROW_IF_FAILED(m_d3D12Device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        m_commandAllocator.Get(),
        nullptr,
        IID_PPV_ARGS(m_commandList.ReleaseAndGetAddressOf())));

    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
#if defined (_DEBUG)
    // If the project is in a debug build, then enable debugging via DirectML debug layers with this flag.
    dmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif
    THROW_IF_FAILED(DMLCreateDevice(
        m_d3D12Device.Get(),
        dmlCreateDeviceFlags,
        IID_PPV_ARGS(m_dmlDevice.GetAddressOf())));

    DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT fp16Query = { DML_TENSOR_DATA_TYPE_FLOAT16 };
    DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT fp16Supported = {};
    THROW_IF_FAILED(m_dmlDevice->CheckFeatureSupport(DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(fp16Query), &fp16Query, sizeof(fp16Supported), &fp16Supported));
    if (fp16Supported.IsSupported)
    {
        std::wcout << L"FP16 is supported." << std::endl;
    }
    else
    {
        std::wcout << L"FP16 is not supported." << std::endl;
    }

    for (int i = DML_TENSOR_DATA_TYPE_UNKNOWN; i <= DML_TENSOR_DATA_TYPE_INT64; ++i) {
        DML_TENSOR_DATA_TYPE type = static_cast<DML_TENSOR_DATA_TYPE>(i);
        std::cout << "Enum value: " << type << std::endl;
    }

}

void DirectMLProcessor::CloseExecuteResetWait()
{
    THROW_IF_FAILED(m_commandList->Close());

    ID3D12CommandList* commandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
    
    ComPtr<ID3D12Fence> d3D12Fence;
    THROW_IF_FAILED(m_d3D12Device->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(d3D12Fence.GetAddressOf())));

    wil::unique_handle fenceEventHandle(::CreateEvent(nullptr, true, false, nullptr));
    THROW_LAST_ERROR_IF_NULL(fenceEventHandle);

    THROW_IF_FAILED(m_commandQueue->Signal(d3D12Fence.Get(), 1));
    THROW_IF_FAILED(d3D12Fence->SetEventOnCompletion(1, fenceEventHandle.get()));

    ::WaitForSingleObjectEx(fenceEventHandle.get(), INFINITE, FALSE);
    
    THROW_IF_FAILED(m_commandAllocator->Reset());
    THROW_IF_FAILED(m_commandList->Reset(m_commandAllocator.Get(), nullptr));
}



void DirectMLProcessor::DoElementWiseAdd(float a, float b)
{
    constexpr UINT tensorSizes[4] = { 1, 2, 3, 4 };
    constexpr UINT tensorElementCount = tensorSizes[0] * tensorSizes[1] * tensorSizes[2] * tensorSizes[3];
    std::wcout << L"tensor element count: " << tensorElementCount << std::endl;

    dml::Graph graph(m_dmlDevice.Get());
    dml::TensorDesc::Dimensions dimensions(std::begin(tensorSizes), std::end(tensorSizes));
    dml::TensorDesc desc = { DML_TENSOR_DATA_TYPE_FLOAT32, dimensions};
    dml::Expression input = dml::InputTensor(graph, 0, desc);
    dml::Expression input2 = dml::InputTensor(graph, 1, desc);
    dml::Expression output = dml::Add(input, input2);

    ComPtr<IDMLCompiledOperator> dmlCompiledOperator;
    DML_EXECUTION_FLAGS executionFlags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
    dmlCompiledOperator.Attach(graph.Compile(executionFlags, { output }).Detach());

    // 24 elements * 4 == 96 bytes.
    UINT64 tensorBufferSize{ desc.totalTensorSizeInBytes };

    ComPtr<IDMLOperatorInitializer> dmlOpInitializer;
    IDMLCompiledOperator* dmlCompiledOperators[] = { dmlCompiledOperator.Get() };
    THROW_IF_FAILED(m_dmlDevice->CreateOperatorInitializer(
        ARRAYSIZE(dmlCompiledOperators),
        dmlCompiledOperators,
        IID_PPV_ARGS(dmlOpInitializer.GetAddressOf())));

    DML_BINDING_PROPERTIES initializeBindingProperties = dmlOpInitializer->GetBindingProperties();
    DML_BINDING_PROPERTIES executeBindingProperties = dmlCompiledOperator->GetBindingProperties();
    UINT descriptorCount = std::max(
        initializeBindingProperties.RequiredDescriptorCount,
        executeBindingProperties.RequiredDescriptorCount);
    
    // Create descriptor heaps.
    ComPtr<ID3D12DescriptorHeap> descriptorHeap;

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = descriptorCount;
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    THROW_IF_FAILED(m_d3D12Device->CreateDescriptorHeap(
        &descriptorHeapDesc,
        IID_PPV_ARGS(descriptorHeap.GetAddressOf())));

    // Set the descriptor heap(s).
    ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = { descriptorHeap.Get() };
    m_commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);

    // Create a binding table over the descriptor heap we just created.
    DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
    dmlBindingTableDesc.Dispatchable = dmlOpInitializer.Get();
    dmlBindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    dmlBindingTableDesc.SizeInDescriptors = descriptorCount;

    ComPtr<IDMLBindingTable> dmlBindingTable;
    THROW_IF_FAILED(m_dmlDevice->CreateBindingTable(
        &dmlBindingTableDesc,
        IID_PPV_ARGS(dmlBindingTable.GetAddressOf())));

    UINT64 temporaryResourceSize = std::max(
        initializeBindingProperties.TemporaryResourceSize,
        executeBindingProperties.TemporaryResourceSize);

    ComPtr<ID3D12Resource> temporaryBuffer;
    if (temporaryResourceSize != 0)
    {
        THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(temporaryBuffer.GetAddressOf())));

        if (initializeBindingProperties.TemporaryResourceSize != 0)
        {
            DML_BUFFER_BINDING bufferBinding{ temporaryBuffer.Get(), 0, temporaryResourceSize };
            DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
            dmlBindingTable->BindTemporaryResource(&bindingDesc);
        }
    }

    UINT64 persistentResourceSize = executeBindingProperties.PersistentResourceSize;
    std::wcout << L"presistentResourceSize: " << L' ' << persistentResourceSize << std::endl;
    ComPtr<ID3D12Resource> persistentBuffer;
    if (persistentResourceSize != 0)
    {
        THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(persistentResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
            D3D12_RESOURCE_STATE_COMMON,
            nullptr,
            IID_PPV_ARGS(persistentBuffer.GetAddressOf())));

        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        DML_BUFFER_BINDING bufferBinding{ persistentBuffer.Get(), 0, persistentResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindOutputs(1, &bindingDesc);
    }

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    ComPtr<IDMLCommandRecorder> dmlCommandRecorder;
    THROW_IF_FAILED(m_dmlDevice->CreateCommandRecorder(
        IID_PPV_ARGS(dmlCommandRecorder.GetAddressOf())));

    // Record execution of the operator initializer.
    dmlCommandRecorder->RecordDispatch(
        m_commandList.Get(),
        dmlOpInitializer.Get(),
        dmlBindingTable.Get());

    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could
    // in principle record the execution into the same command list as the initialization, but you need only to Initialize
    // once, and typically you want to Execute an operator more frequently than that.
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
        DML_BUFFER_BINDING bufferBinding{ temporaryBuffer.Get(), 0, temporaryResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindTemporaryResource(&bindingDesc);
    }

    if (persistentResourceSize != 0)
    {
        DML_BUFFER_BINDING bufferBinding{ persistentBuffer.Get(), 0, persistentResourceSize };
        DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
        dmlBindingTable->BindPersistentResource(&bindingDesc);
    }

    // Create tensor buffers for upload/input/output/readback of the tensor elements.

    ComPtr<ID3D12Resource> uploadBuffer;
    THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(uploadBuffer.GetAddressOf())));

    ComPtr<ID3D12Resource> uploadBuffer2;
    THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(uploadBuffer2.GetAddressOf())));

    ComPtr<ID3D12Resource> inputBuffer;
    THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(inputBuffer.GetAddressOf())));

    ComPtr<ID3D12Resource> inputBuffer2;
    THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(inputBuffer2.GetAddressOf())));

    std::wcout << std::fixed; std::wcout.precision(4);
    std::array<FLOAT, tensorElementCount> inputTensorElementArray, inputTensorElementArray2;
    {
        std::wcout << L"input tensor: ";
        for (auto & element : inputTensorElementArray)
        {
            element = a;
            std::wcout << element << L' ';
        };
        std::wcout << std::endl;
        
        std::wcout << L"input tensor2: ";
        for (auto & element : inputTensorElementArray2)
        {
            element = b;
            std::wcout << element << L' ';
        };
        std::wcout << std::endl;

        D3D12_SUBRESOURCE_DATA tensorSubresourceData{};
        tensorSubresourceData.pData = inputTensorElementArray.data();
        tensorSubresourceData.RowPitch = static_cast<LONG_PTR>(tensorBufferSize);
        tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

        D3D12_SUBRESOURCE_DATA tensorSubresourceData2{};
        tensorSubresourceData2.pData = inputTensorElementArray2.data();
        tensorSubresourceData2.RowPitch = static_cast<LONG_PTR>(tensorBufferSize);
        tensorSubresourceData2.SlicePitch = tensorSubresourceData2.RowPitch;
        // Upload the input tensor to the GPU.
        ::UpdateSubresources(
            m_commandList.Get(),
            inputBuffer.Get(),
            uploadBuffer.Get(),
            0,
            0,
            1,
            &tensorSubresourceData);

        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                inputBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            )
        );

        ::UpdateSubresources(
            m_commandList.Get(),
            inputBuffer2.Get(),
            uploadBuffer2.Get(),
            0,
            0,
            1,
            &tensorSubresourceData2);

        m_commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                inputBuffer2.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            )
        );
    }

    DML_BUFFER_BINDING inputBufferBinding{ inputBuffer.Get(), 0, tensorBufferSize };
    DML_BINDING_DESC inputBindingDesc{ DML_BINDING_TYPE_BUFFER, &inputBufferBinding };
    DML_BUFFER_BINDING inputBufferBinding2{ inputBuffer2.Get(), 0, tensorBufferSize };
    DML_BINDING_DESC inputBindingDesc2{ DML_BINDING_TYPE_BUFFER, &inputBufferBinding2 };
    
    DML_BINDING_DESC inputBindingDescs[] = { inputBindingDesc, inputBindingDesc2 };
    dmlBindingTable->BindInputs(2, inputBindingDescs);

    ComPtr<ID3D12Resource> outputBuffer;
    THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(outputBuffer.GetAddressOf())));

    DML_BUFFER_BINDING outputBufferBinding{ outputBuffer.Get(), 0, tensorBufferSize };
    DML_BINDING_DESC outputBindingDesc{ DML_BINDING_TYPE_BUFFER, &outputBufferBinding };
    dmlBindingTable->BindOutputs(1, &outputBindingDesc);

    // Record execution of the compiled operator.
    dmlCommandRecorder->RecordDispatch(m_commandList.Get(), dmlCompiledOperator.Get(), dmlBindingTable.Get());

    CloseExecuteResetWait();

    // The output buffer now contains the result of the identity operator,
    // so read it back if you want the CPU to access it.

    ComPtr<ID3D12Resource> readbackBuffer;
    THROW_IF_FAILED(m_d3D12Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(tensorBufferSize),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(readbackBuffer.GetAddressOf())));

    m_commandList->ResourceBarrier(
        1,
        &CD3DX12_RESOURCE_BARRIER::Transition(
            outputBuffer.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE
        )
    );

    m_commandList->CopyResource(readbackBuffer.Get(), outputBuffer.Get());

    CloseExecuteResetWait();

    D3D12_RANGE tensorBufferRange{ 0, static_cast<SIZE_T>(tensorBufferSize) };
    FLOAT* outputBufferData{};
    THROW_IF_FAILED(readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void**>(&outputBufferData)));

    std::wstring outputString = L"output tensor: ";
    for (size_t tensorElementIndex{ 0 }; tensorElementIndex < tensorElementCount; ++tensorElementIndex, ++outputBufferData)
    {
        outputString += std::to_wstring(*outputBufferData) + L' ';
    }

    std::wcout << outputString << std::endl;
    OutputDebugStringW(outputString.c_str());

    D3D12_RANGE emptyRange{ 0, 0 };
    readbackBuffer->Unmap(0, &emptyRange);
}
