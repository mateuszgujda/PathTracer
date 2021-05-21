#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

__global__ void hittable_list_gpu(hittable** d_world, hittable** d_list, int list_size);
__global__ void copy_hittable_to_list_gpu(hittable** d_obj, hittable** d_list, int index);

class hittable_list : public hittable {
	public :
		__host__ __device__ hittable_list() {}
		__host__ __device__ hittable_list(hittable** list, int list_size) {
			this->list = list;
			this->list_size = list_size;
		}
		__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

		__host__ virtual void create_hittable_on_gpu() override {
			checkCudaErrors(cudaMalloc(&d_this, sizeof(hittable_list*)));
			checkCudaErrors(cudaMalloc(&d_list, list_size * sizeof(hittable*)));
			hittable_list_gpu << <1, 1 >> > (d_this, d_list, list_size);
		}

		__host__ void copy_list_to_gpu() {
			if (list_size > 0 && d_list != NULL && d_this != NULL) {
				for (int i = 0; i < list_size; i++) {
					copy_hittable_to_list_gpu << <1, 1 >> > (d_list, list[i]->d_this, i);
				}
			}
		}

		__host__ __device__ ~hittable_list(){
			#if !defined(__CUDA_ARCH__)
				if (d_list != NULL) {
					checkCudaErrors(cudaFree(d_list));
				}
			#endif
		}

	public:
		hittable** list = NULL;
		hittable** d_list = NULL;
		int list_size = 0;
};

__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	hit_record temp_rec;
	bool hit_anything = false;
	auto closest_so_far = t_max;

	for (int i = 0; i < list_size; i++) {
		if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}

	return hit_anything;
}

__global__ void hittable_list_gpu(hittable** d_world, hittable** d_list, int list_size) {
 	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_world = new hittable_list(d_list, list_size);
	}
}

__global__ void copy_hittable_to_list_gpu(hittable** d_list, hittable** d_obj, int index) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_list[index] = *d_obj;
	}
}


#endif