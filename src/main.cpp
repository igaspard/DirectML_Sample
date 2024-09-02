#include "DirectMLProcessor.hpp"

int main(int argc, char const *argv[])
{
    DirectMLProcessor helloDML;
    uint32_t shapes[] = {1, 1, 8, 1};
    float data0[] = {0.0202845, 0.704157, 0.12591, 0.101374, 0.863722, -0.915456, 0.333993, -0.123652};
    float data1[] = {-0.29151, -0.623414, -0.913637, -0.737006, 0.144149, -0.834521, -0.509416, 0.899506};
    float zeroArray[8] = {0.0f};

    helloDML.SetTensorData("add0", shapes, DML_TENSOR_DATA_TYPE_FLOAT32, data0, sizeof(data0));
    helloDML.SetTensorData("add1", shapes, DML_TENSOR_DATA_TYPE_FLOAT32, data1, sizeof(data1));
    helloDML.SetTensorData("dst", shapes, DML_TENSOR_DATA_TYPE_FLOAT32, zeroArray, sizeof(zeroArray));
    helloDML.ElementWiseAdd("add0", "add1", "dst");

    float result[8];
    helloDML.GetTensorData("dst", shapes, DML_TENSOR_DATA_TYPE_FLOAT32, result, sizeof(result));
    for (int i = 0; i < 8; i++)
    {
        printf("%f\n", result[i]);
        if (result[i] != data0[i] + data1[i])
        {
            printf("Error: %f != %f + %f\n", result[i], data0[i], data1[i]);
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

    return 0;
}
