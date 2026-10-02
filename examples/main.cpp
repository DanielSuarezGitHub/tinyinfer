#include <iostream>

void cuda_smoke_test();

int main() {
    std::cout << "TinyInfer starting...\n";

    cuda_smoke_test();

    std::cout << "CUDA kernel launched successfully!\n";
    return 0;
}