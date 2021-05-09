
#include "commons.h"
#include <curand_kernel.h>
#include <iostream>
#include<time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "sphere.h"
#include "material.h"
#include "camera.h"
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

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__device__ color ray_color(const ray& r, const hittable** world, curandState* local_rand_state) {
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
    ray cur_ray = r;
    for (int i = 0; i < 50; i++) {
        hit_record hit;
        if ((*world)->hit(cur_ray, 0.001f, INFINITY, hit)) {
            ray scattered;
            vec3 attenuation;
            if (hit.material->scatter(cur_ray, hit, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * color(1.0f, 1.0f, 1.0f) + t * color(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }
    }
    // exceeded recursion value
    return color(0.0f, 0.0f, 0.0f);
}

__global__ void create_world(hittable** d_list, hittable** d_world, camera** d_camera, float aspect_ratio) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, new lambertian(vec3(0.1f, 0.2f, 0.5f)));
        d_list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f, new lambertian(vec3(0.8f, 0.8f, 0.0f)));
        d_list[2] = new sphere(vec3(1.0f, 0.0f, -1.0f), 0.5f, new metal(vec3(0.8f, 0.6f, 0.2f), 1.0f));
        d_list[3] = new sphere(vec3(-1.0f, 0.0f, -1.0f), 0.5f, new dielectric(1.5f));
        d_list[4] = new sphere(vec3(-1.0f, 0.0f, -1.0f), -0.4f, new dielectric(1.5f));
        *d_world = new hittable_list(d_list, 5);
        vec3 lookfrom(3.0f, 3.0f, 2.0f);
        vec3 lookat(0.0f, 0.0f, -1.0f);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 2.0f;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0.0f, 1.0f, 0.0f),
            20.0f,
            aspect_ratio,
            aperture,
            dist_to_focus);
    }
}

__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    hittable_list* list = (hittable_list*)d_world;

    for (int i = 0; i < list->list_size; i++) {
        delete ((sphere*)d_list[i])->material_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

__global__ void render(vec3* fb, int max_x, int max_y, int samples_per_pixel, camera** camera, hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    color col = color(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < samples_per_pixel; k++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*camera)->get_ray(u, v);
        col += ray_color(r, world, &local_rand_state);
    }
    col /= float(samples_per_pixel);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

int main() {
    //Image
    const float aspect_ratio = 16.0f / 9.0f;
    const int image_width = 800;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    int tx = 8;
    int ty = 8;
    int samples_per_pixel = 100;
    int num_of_spheres = 5;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    //Camera
    //on gpu

    //Render

    // allocate memory for pixels
    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(color);
    color* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    //Cuda Randomizer
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // allocate Memory for list of objects
    hittable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_of_spheres * sizeof(hittable*)));
    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world<<< 1, 1 >>>(d_list, d_world, d_camera, aspect_ratio);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());



    //Start clock
    clock_t start, cuda_stop, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (fb, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state);
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
            int ir = static_cast<int>(256 * clamp(pixelColor.r(), 0.0, 0.999));
            int ig = static_cast<int>(256 * clamp(pixelColor.g(), 0.0, 0.999));
            int ib = static_cast<int>(256 * clamp(pixelColor.b(), 0.0, 0.999));
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
    free_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
    return 0;
}