
#include "commons.h"
#include <iostream>
#include<time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "sphere.h"
#include "hittable_list.h"
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

__device__ color ray_color(const ray& r, const hittable** world) {
    hit_record hit;
    if ((*world)->hit(r, 0, INFINITY, hit)) {
        return 0.5 * (hit.normal + color(1, 1, 1));
    }
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
}

__global__ void create_world(hittable** d_list, hittable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new sphere(vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
        *d_world = new hittable_list(d_list, 2);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
}

__global__ void render(vec3* fb, int max_x, int max_y, point3 lower_left_corner, vec3 horizontal, vec3 vertical, point3 origin, hittable** world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    fb[pixel_index] = ray_color(r, world);
}

int main() {
    //Image
    const float aspect_ratio = 16.0f / 9.0f;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    //Camera
    float viewport_height = 2.0f;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0f;

    point3 origin = point3(0, 0, 0);
    vec3 horizontal = vec3(viewport_width, 0, 0);
    vec3 vertical = vec3(0, viewport_height, 0);
    point3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

    //Render

    // allocate Memory for list of objects
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    create_world<<< 1, 1 >>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // allocate memory for pixels
    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(color);
    color* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));


    //Start clock
    clock_t start, cuda_stop, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render << <blocks, threads >> > (fb, image_width, image_height, lower_left_corner, horizontal, vertical, origin, d_world);
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
 
    //Freeing memory
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
    return 0;
}