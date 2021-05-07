#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include<time.h>
#include "vec3.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(color* fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    fb[pixel_index] = color(float(i) / max_x, float(j) / max_y, 0.2f);
}

int main() {
    int image_width = 1200;
    int image_height = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(color);

    // allocate FB
    color* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    clock_t start, cuda_stop, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render << <blocks, threads >> > (fb, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cuda_stop = clock();
    double timer_seconds = ((double)(cuda_stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Cuda computation took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    uint8_t* pixels = new uint8_t[image_width * image_height * 3];
    int write_index = 0;
    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            color pixelColor = fb[pixel_index];
            int ir = int(255.99 * pixelColor.x());
            int ig = int(255.99 * pixelColor.y());
            int ib = int(255.99 * pixelColor.z());
            pixels[write_index++] = ir;
            pixels[write_index++] = ig;
            pixels[write_index++] = ib;
        }
    }

    stbi_write_bmp("image.bmp", image_width, image_height, 3, pixels);
    stop = clock();
    timer_seconds = ((double)(stop - cuda_stop)) / CLOCKS_PER_SEC;
    std::cerr << "Image write took " << timer_seconds << " seconds. \n";
 
    checkCudaErrors(cudaFree(fb));


}