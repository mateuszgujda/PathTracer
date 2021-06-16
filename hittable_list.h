#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

__global__ void hittable_list_gpu(hittable** d_world, hittable** d_list, int list_size);
__global__ void copy_hittable_to_list_gpu(hittable** d_obj, hittable** d_list, int index);
__global__ void free_hittable_list_gpu(hittable** d_list);

/**
 * Klasa zawierajπca listÍ obiektÛw, ktÛre mogπ wystÍpowaÊ w scene.
 */
class hittable_list : public hittable {
	public :
		/**
		 * Konstruktor.
		 * 
		 * \return 
		 */
		__host__ __device__ hittable_list() {}
		/**
		 * Konstruktor.
		 * 
		 * \param list lista obiektÛw do przypisania
		 * \param list_size rozmiar listy
		 * \return 
		 */
		__host__ __device__ hittable_list(hittable** list, int list_size) {
			this->list = list;
			this->list_size = list_size;
		}
		//! @copydoc hittable::hit(r,t_min,t_max,rec)
		__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
		//! @copydoc hittable::bounding_box(time0,time1,output_box)
		__device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

		//! @copydoc hittable::create_hittable_on_gpu()
		__host__ virtual void create_hittable_on_gpu() override {
			checkCudaErrors(cudaMalloc(&d_this, sizeof(hittable_list*)));
			checkCudaErrors(cudaMalloc(&d_list, list_size * sizeof(hittable*)));
			hittable_list_gpu << <1, 1 >> > (d_this, d_list, list_size);
		}
		
		/**
		 * Przypisywanie obiektÛw do listy na GPU
		 * 
		 * \return 
		 */
		__host__ void copy_list_to_gpu() {
			if (list_size > 0 && d_list != NULL && d_this != NULL) {
				for (int i = 0; i < list_size; i++) {
					copy_hittable_to_list_gpu << <1, 1 >> > (d_list, list[i]->d_this, i);
				}
			}
		}

		/**
		 * Destruktor.
		 * 
		 * \return 
		 */
		__host__ __device__ ~hittable_list(){
			#if !defined(__CUDA_ARCH__)
				if (d_list != NULL) {
					free_hittable_list_gpu << <1, 1 >> > (d_list);
					checkCudaErrors(cudaFree(d_list));
				}
			#endif
		}

	public:
		/**
		 * Lista obiektÛw.
		 */
		hittable** list = NULL;
		/**
		 * Wskaünik na obiekt GPU.
		 */
		hittable** d_list = NULL;
		/**
		 * Rozmiar listy.
		 */
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

__device__ bool hittable_list::bounding_box(float t0, float t1, aabb& box) const {
	if (list_size < 1) return false;
	aabb temp_box;
	bool first_true = list[0]->bounding_box(t0, t1, temp_box);
	if (!first_true)
		return false;
	else
		box = temp_box;
	for (int i = 1; i < list_size; i++) {
		if (list[i]->bounding_box(t0, t1, temp_box)) {
			box = surrounding_box(box, temp_box);
		}
		else
			return false;
	}
	return true;
}



/**
 * Tworzenie \see hittable_list na GPU.
 * 
 * \param d_world wskaünik na obiekt
 * \param d_list wskaünik do listy obiektÛw
 * \param list_size rozmiar listy
 * \return 
 */
__global__ void hittable_list_gpu(hittable** d_world, hittable** d_list, int list_size) {
 	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*d_world = new hittable_list(d_list, list_size);
	}
}

/**
 * Kopiowanie obiektu do listy na GPU.
 * 
 * \param d_list wskaünik do listy obiektÛw
 * \param d_obj wskaünik na obiekt do dodania do listy
 * \param index miejce na liúcie
 * \return 
 */
__global__ void copy_hittable_to_list_gpu(hittable** d_list, hittable** d_obj, int index) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		d_list[index] = *d_obj;
	}
}

/**
 * Zwolnienie listy z GPU.
 * 
 * \param d_list Wskaünik na liste
 * \return 
 */
__global__ void free_hittable_list_gpu(hittable** d_list) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		delete* d_list;
	}
}


#endif