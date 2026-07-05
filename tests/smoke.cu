#include <cstdio>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(err__));                \
            return 1;                                                         \
        }                                                                     \
    } while (0)

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    std::printf("visible_devices=%d\n", device_count);
    if (device_count < 4) {
        std::fprintf(stderr, "need at least 4 visible CUDA devices\n");
        return 1;
    }

    for (int dev = 0; dev < 4; ++dev) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::printf("device%d=%s pci=%04x:%02x:%02x\n", dev, prop.name,
                    prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    }

    for (int src = 0; src < 4; ++src) {
        for (int dst = 0; dst < 4; ++dst) {
            if (src == dst) {
                continue;
            }
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, src, dst));
            std::printf("p2p %d -> %d = %d\n", src, dst, can_access);
        }
    }

    for (int dev = 0; dev < 4; ++dev) {
        CUDA_CHECK(cudaSetDevice(dev));
        CUDA_CHECK(cudaFree(nullptr));
    }

    std::puts("cuda_smoke=PASS");
    return 0;
}
