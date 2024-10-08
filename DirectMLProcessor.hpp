#pragma once

#define NOMINMAX

#include <d3dx12.h>
#include <dxgi1_4.h>
#include <wil/resource.h>
#include <stdexcept>
#include <DirectML.h>
#include <DirectMLX.h>
#include <string>
#include <unordered_map>

struct TensorInfo
{
    uint64_t bufferSize;
    uint32_t elementCount;
    dml::TensorDimensions dimensions;
    dml::TensorDesc desc;

    Microsoft::WRL::ComPtr<ID3D12Resource> resource;
    Microsoft::WRL::ComPtr<ID3D12Resource> uploadResource;
};

class DirectMLProcessor
{
  public:
    DirectMLProcessor(std::string adapterNameFilter = "NPU")
    {
        InitializeDirectML(adapterNameFilter);
    } // Constructor

    ~DirectMLProcessor(); // Destructor

    void SetTensorData(std::string name, uint32_t *shapes, DML_TENSOR_DATA_TYPE type, const void *data, size_t size);
    void GetTensorData(std::string name, uint32_t *shapes, DML_TENSOR_DATA_TYPE type, void *data, size_t size);

    void ElementWiseAddBcast(std::string src0, std::string src1, std::string dst);

    void FreeResources();

  private:
    void InitializeDirectML(std::string adapterNameFilter);
    void CloseExecuteResetWait();

    bool CanBroadcast(const dml::TensorDimensions &a, const dml::TensorDimensions &b);
    dml::Expression ReshapeAndBroadcastTensor(dml::Expression originalTensor, const dml::TensorDimensions &targetShape);

    Microsoft::WRL::ComPtr<ID3D12Device> m_d3D12Device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
    Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;

    std::unordered_map<std::string, TensorInfo *> m_tensorInfoMap;
};
;
