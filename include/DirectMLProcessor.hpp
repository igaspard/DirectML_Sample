#pragma once

#define NOMINMAX

#include "d3dx12.h"
#include <dxgi1_4.h>
#include <wil/resource.h>

#include <DirectML.h> // The DirectML header from the Windows SDK.
#include <DirectMLX.h>
#include <string>
#include <unordered_map>

class DirectMLProcessor
{
  public:
    DirectMLProcessor(bool forceNpu = true) : m_tensorBufferSize(0), m_tensorElementCount(0), m_tensorSizes{1, 1, 1, 1}
    {
        InitializeDirectML(forceNpu);
    } // Constructor

    ~DirectMLProcessor(); // Destructor

    void SetTensorData(std::string tensor_name, uint32_t *shapes, DML_TENSOR_DATA_TYPE type, const void *data,
                       size_t size);
    void GetTensorData(std::string tensor_name, uint32_t *shapes, DML_TENSOR_DATA_TYPE type, void *data, size_t size);

    void ElementWiseAdd(std::string src0, std::string src1, std::string dst);

  private:
    Microsoft::WRL::ComPtr<ID3D12Device> m_d3D12Device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
    Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;

    void InitializeDirectML(bool forceNpu);
    void CloseExecuteResetWait();

    uint64_t m_tensorBufferSize;
    uint32_t m_tensorElementCount;
    uint32_t m_tensorSizes[4];
    dml::TensorDesc m_desc;

    std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3D12Resource>> m_tensorResourceMap;
    std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3D12Resource>> m_tensorUploadResourceMap;

    static const size_t c_numOutputs = 2;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_outputBuffer[c_numOutputs];
};
;
