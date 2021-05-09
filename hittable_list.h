#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

class hittable_list : public hittable {
	public :
		__device__ hittable_list() {}
		__device__ hittable_list(hittable** list, int list_size) {
			this->list = list;
			this->list_size = list_size;
		}
		__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

	public:
		hittable** list = NULL;
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


#endif