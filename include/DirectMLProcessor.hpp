#pragma once

#define NOMINMAX

#include <wil/resource.h>
#include "d3dx12.h"
#include <dxgi1_4.h>

#include <DirectML.h> // The DirectML header from the Windows SDK.
#include <DirectMLX.h>

class DirectMLProcessor {
public:
    DirectMLProcessor(bool forceNpu = true);  // Constructor
    ~DirectMLProcessor(); // Destructor

    void SetTensorData(const void * data, size_t size);
    void DoElementWiseAdd(float a, float b);
private:
    Microsoft::WRL::ComPtr<ID3D12Device> m_d3D12Device;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> m_commandQueue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> m_commandAllocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> m_commandList;
    Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;

    void InitializeDirectML(bool forceNpu);
    void CloseExecuteResetWait();
    
};