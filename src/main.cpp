#include "DirectMLProcessor.hpp"

int main(int argc, char const *argv[])
{
    DirectMLProcessor helloDML;

    helloDML.DoElementWiseAdd(1.0f, 2.0f);
    return 0;
}
