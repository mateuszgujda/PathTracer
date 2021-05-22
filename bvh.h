#ifndef BVH_H
#define BVH_H

#include "commons.h"

#include <thrust/sort.h>
#include "hittable.h"
#include "hittable_list.h"


//class bvh_node : public hittable {
//public:
//    bvh_node();
//
//    __host__ __device__ bvh_node(const hittable_list& list, float time0, float time1)
//        : bvh_node(list.list, 0, list.list_size, time0, time1)
//    {}
//
//    __host__ __device__ bvh_node(
//        const hittable** src_objects,
//        size_t start, size_t end, float time0, float time1);
//
//     __device__ virtual bool hit(
//        const ray& r, float t_min, float t_max, hit_record& rec) const override;
//
//     __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;
//
//public:
//    hittable* left;
//    hittable* right;
//    aabb box;
//};
//
//__device__ bool bvh_node::bounding_box(float time0, float time1, aabb& output_box) const {
//    output_box = box;
//    return true;
//}
//
//__device__  __host__ bvh_node::bvh_node(
//    hittable** src_objects,
//    size_t start, size_t end, float time0, float time1
//) {
//    auto objects = src_objects; // Create a modifiable array of the source scene objects
//
//    int axis = random_int(0, 2);
//    auto comparator = (axis == 0) ? box_x_compare
//        : (axis == 1) ? box_y_compare
//        : box_z_compare;
//
//    size_t object_span = end - start;
//
//    if (object_span == 1) {
//        left = right = objects[start];
//    }
//    else if (object_span == 2) {
//        if (comparator(objects[start], objects[start + 1])) {
//            left = objects[start];
//            right = objects[start + 1];
//        }
//        else {
//            left = objects[start + 1];
//            right = objects[start];
//        }
//    }
//    else {
//        thrust::sort(objects[start], objects[end], comparator);
//
//        auto mid = start + object_span / 2;
//        left = bvh_node(objects, start, mid, time0, time1);
//        right = bvh_node(objects, mid, end, time0, time1);
//    }
//
//    aabb box_left, box_right;
//
//    if (!left->bounding_box(time0, time1, box_left)
//        || !right->bounding_box(time0, time1, box_right)
//        )
//        std::cerr << "No bounding box in bvh_node constructor.\n";
//
//    box = surrounding_box(box_left, box_right);
//}
//
//
//__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
//    if (!box.hit(r, t_min, t_max))
//        return false;
//
//    bool hit_left = left->hit(r, t_min, t_max, rec);
//    bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);
//
//    return hit_left || hit_right;
//}

#endif