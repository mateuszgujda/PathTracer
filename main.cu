
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
#include "scene.h"
#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TBP 512;


__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984+ pixel_index, 0, 0, &rand_state[pixel_index]);
}

__device__ color ray_color(const ray& r, const color& background, const hittable** world, curandState* local_rand_state,  int depth) {
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0, 0, 0);

    // If the ray hits nothing, return the background color.
    if (!(*world)->hit(r, 0.001f, INFINITY, rec))
        return background;

    ray scattered;
    color attenuation;
    color emitted = rec.material->emitted(rec.p);

    if (!rec.material->scatter(r, rec, attenuation, scattered, local_rand_state))
        return emitted;

    return emitted + attenuation * ray_color(scattered, background, world, local_rand_state, depth - 1);
}


__global__ void free_world(hittable** d_list, hittable** d_world, camera** d_camera) {
    hittable_list* list = (hittable_list*)d_world;

    for (int i = 0; i < list->list_size; i++) {
        if (d_list[i]->material_ptr != NULL) {
            delete d_list[i]->material_ptr;
        }
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}



__global__ void render(vec3* fb, int max_x, int max_y, int samples_per_pixel, color background, camera** camera, hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    color col = color(0.0f, 0.0f, 0.0f);
    int blocks = samples_per_pixel / TBP + 1;
    for (int k = 0; k < samples_per_pixel; k++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*camera)->get_ray(u, v, &local_rand_state);
        col += ray_color(r, background, world, &local_rand_state, 5);
    }
    col /= float(samples_per_pixel);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

int main() {
    //Image
   // const float aspect_ratio = 16.0f / 9.0f;
    const float aspect_ratio = 1;
    const int image_width = 600;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    int tx = 8;
    int ty = 8;
    int samples_per_pixel = 80;
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


    scene* sc = new scene(2);
    //Start clock
    clock_t start, cuda_stop, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (fb, image_width, image_height, samples_per_pixel,sc->background_color, sc->cam->d_this, sc->world->d_this, d_rand_state);
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
    free_world << <1, 1 >> > (sc->world->d_list, sc->world->d_this, sc->cam->d_this);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));
    delete sc;


    cudaDeviceReset();
    return 0;
}