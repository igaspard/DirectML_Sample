#include "DirectMLProcessor.hpp"

int main(int argc, char const *argv[])
{
    std::string adapterNameFilter = (argc > 1) ? argv[1] : "NPU";
    DirectMLProcessor helloDML(adapterNameFilter);
    uint32_t shapes[] = {1, 1, 8, 1};
    float data0[] = {0.0202845, 0.704157, 0.12591, 0.101374, 0.863722, -0.915456, 0.333993, -0.123652};
    float data1[] = {-0.29151, -0.623414, -0.913637, -0.737006, 0.144149, -0.834521, -0.509416, 0.899506};
    float zeroArray[8] = {0.0f};

    helloDML.SetTensorData("add0", shapes, DML_TENSOR_DATA_TYPE_FLOAT32, data0, sizeof(data0));
    helloDML.SetTensorData("add1", shapes, DML_TENSOR_DATA_TYPE_FLOAT32, data1, sizeof(data1));
    helloDML.SetTensorData("dst", shapes, DML_TENSOR_DATA_TYPE_FLOAT32, zeroArray, sizeof(zeroArray));
    helloDML.ElementWiseAddBcast("add0", "add1", "dst");

    float result[8];
    helloDML.GetTensorData("dst", shapes, DML_TENSOR_DATA_TYPE_FLOAT32, result, sizeof(result));
    for (int i = 0; i < 8; i++)
    {
        printf("%f\n", result[i]);
        if (result[i] != data0[i] + data1[i])
        {
            printf("Error: %f != %f + %f, expected %f\n", result[i], data0[i], data1[i], data0[i] + data1[i]);
            return 1;
        }
    }
    printf("data0 + data1 is equal to result\n");

    float data2[1024];
    for (float i = 0.0f; i < 1024.0f; i++)
        data2[(int)i] = i;
    uint32_t shapes2[] = {1, 1, 1024, 1};
    helloDML.SetTensorData("sent0", shapes2, DML_TENSOR_DATA_TYPE_FLOAT32, data2, sizeof(data2));
    float result2[1024] = {0.0f};
    helloDML.GetTensorData("sent0", shapes2, DML_TENSOR_DATA_TYPE_FLOAT32, result2, sizeof(result2));
    for (int i = 0; i < 1024; i++)
    {
        if (result2[i] != data2[i])
        {
            printf("Error: %f != %f\n", result2[i], data2[i]);
            return 1;
        }
    }
    printf("data2 is equal to result2\n");


    float data3[1] = {1.0f};
    uint32_t shapes3[] = {1, 1, 1, 1};
    helloDML.SetTensorData("add2", shapes3, DML_TENSOR_DATA_TYPE_FLOAT32, data3, sizeof(data3));
    helloDML.ElementWiseAddBcast("add0", "add2", "dst");

    helloDML.GetTensorData("dst", shapes, DML_TENSOR_DATA_TYPE_FLOAT32, result, sizeof(result));
    for (int i = 0; i < 8; i++)
    {
        printf("%f\n", result[i]);
        if (result[i] != data0[i] + data3[0])
        {
            printf("Error: %f != %f + %f\n", result[i], data0[i], data3[0]);
            return 1;
        }
    }
    printf("broadcast add is equal to result\n");

    helloDML.FreeResources();

    return 0;
}
