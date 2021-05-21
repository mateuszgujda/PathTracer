#pragma once
#include "hittable_list.h";
#include "material.h";
#include "sphere.h";
#include "camera.h";
#include "commons.h";
#include <vector>;

class scene {
public:
	scene(float aspect_ratio) {

        auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
        ground_material->create_material_on_gpu();
        auto ground = new sphere(point3(0, -1000, 0), 1000, ground_material);
        hitables.push_back(ground);

        for (int a = -1; a < 1; a++) {
            for (int b = -1; b < 1; b++) {
                auto choose_mat = random_float();
                point3 center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

                if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                    material* sphere_material;
                    sphere* obj;

                    if (choose_mat < 0.8) {
                        // diffuse
                        auto albedo = color(random_float(), random_float(), random_float());
                        sphere_material = new lambertian(albedo);
                        sphere_material->create_material_on_gpu();
                        obj = new sphere(center, 0.2f, sphere_material);
                        
                        hitables.push_back(obj);
                    }
                    else if (choose_mat < 0.95) {
                        // metal
                        auto albedo = color(random_float(0.5f,1.0f), random_float(0.5f, 1.0f), random_float(0.5f, 1.0f));
                        auto fuzz = random_float(0.0f, 0.5f);
                        sphere_material = new metal(albedo, fuzz);
                        obj = new sphere(center, 0.2f, sphere_material);

                        hitables.push_back(obj);
                    }
                    else {
                        // glass
                        sphere_material = new dielectric(1.5f);
                        sphere_material->create_material_on_gpu();
                        obj = new sphere(center, 0.2, sphere_material);

                        hitables.push_back(obj);
                    }
                }
            }
        }

        auto material1 = new dielectric(1.5f);
        material1->create_material_on_gpu();
        auto sph = new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, material1);
        hitables.push_back(sph);
        

        auto material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
        material2->create_material_on_gpu();
        auto sph2 = new sphere(point3(-4, 1, 0), 1.0f, material2);
        hitables.push_back(sph2);

        auto material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
        material3->create_material_on_gpu();
        auto sph3 = new sphere(point3(4, 1, 0), 1.0f, material3);
        hitables.push_back(sph3);

        cudaDeviceSynchronize();
        hittable** list = new hittable* [hitables.size()];
        for (int i = 0; i < hitables.size(); i++) {
            list[i] = hitables[i];
            list[i]->create_hittable_on_gpu();
        }
        cudaDeviceSynchronize();
        world = new hittable_list(list, hitables.size());
        world->create_hittable_on_gpu();
        world->copy_list_to_gpu();
	
        point3 lookfrom(13, 2, 3);
        point3 lookat(0, 0, 0);
        vec3 vup(0, 1, 0);
        auto dist_to_focus = 10.0;
        auto aperture = 0.1;
        cam = new camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
        cam->create_camera_on_gpu();
    
        cudaDeviceSynchronize();
    }

    scene(char* fileName) {

    }

    ~scene() {
        for (int i = 0; i < world->list_size; i++) {
            if (hitables[i]->material_ptr != NULL) {
                delete hitables[i]->material_ptr;
            }
            delete hitables[i];
        }
        delete world->list;
        delete world;
        delete cam;

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
    }



    public :
        std::vector<hittable*> hitables;
        hittable_list* world;
        camera* cam;
};