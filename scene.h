#pragma once
#include "hittable_list.h";
#include "material.h";
#include "sphere.h";
#include "camera.h"
#include "aarect.h";
#include "commons.h";
#include <vector>;
#include "cylinder.h";
#include "cone.h";

class scene {
public:
    scene() {
        auto ground_material = new lambertian(color(0.5, 0.5, 0.5));
        ground_material->create_material_on_gpu();
        auto ground = new sphere(point3(0, -1000, 0), 1000, ground_material);
        hitables.push_back(ground);
        auto material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
        material2->create_material_on_gpu();
        auto sph2 = new sphere(point3(0, 2, 0), 2.0f, material2);
        hitables.push_back(sph2);

        auto difflight = new diffuse_light(color(4, 4, 4));
        difflight->create_material_on_gpu();
        auto rect = new xy_rect(3, 5, 1, 3, -2, difflight);
        hitables.push_back(rect);


        cudaDeviceSynchronize();
        hittable** list = new hittable * [hitables.size()];
        for (int i = 0; i < hitables.size(); i++) {
            list[i] = hitables[i];
            list[i]->create_hittable_on_gpu();
        }
        cudaDeviceSynchronize();
        world = new hittable_list(list, hitables.size());
        world->create_hittable_on_gpu();
        world->copy_list_to_gpu();

        background_color = color(0, 0, 0);
        point3 lookfrom = point3(26, 3, 6);
        point3 lookat = point3(0, 2, 0);
        float vfov = 20.0;

        auto dist_to_focus = 10.0;
        auto aperture = 0.1;
        cam = new camera(lookfrom, lookat, vec3(0,1,0), vfov, 16.0f / 9.0f, aperture, dist_to_focus);
        cam->create_camera_on_gpu();

        cudaDeviceSynchronize();
    }

    scene(int value) {
        auto red = new lambertian(color(.65f, .05f, .05f));
        red->create_material_on_gpu();
        auto wall1 = new yz_rect(0, 555, 0, 555, 0, red);
        hitables.push_back(wall1);

        auto white = new lambertian(color(.73f, .73f, .73f));
        white->create_material_on_gpu();
        auto ceiling = new xz_rect(0, 555, 0, 555, 0, white);
        hitables.push_back(ceiling);
        auto white2 = new lambertian(color(.73f, .73f, .73f));
        white2->create_material_on_gpu();
        auto ceiling2 = new xz_rect(0, 555, 0, 555, 555, white2);
        hitables.push_back(ceiling2);
        auto white3 = new lambertian(color(.73f, .73f, .73f));
        white3->create_material_on_gpu();
        auto ceiling3 = new xy_rect(0, 555, 0, 555, 555, white3);
        hitables.push_back(ceiling3);

        auto green2 = new lambertian(color(.12f, .45f, .15f));
        green2->create_material_on_gpu();
        cylinder2* cylinder = new cylinder2(point3(348, 0, 200),  200, 64, green2);
        hitables.push_back(cylinder);
        
        auto green = new lambertian(color(.12f, .45f, .15f));
        green->create_material_on_gpu();
        auto wall2 = new yz_rect(0, 555, 0, 555, 555, green);
        hitables.push_back(wall2);

        auto light = new diffuse_light(color(15, 15, 15));
        light->create_material_on_gpu();
        auto wall3 = new xz_rect(213, 343, 227, 332, 554, light);
        hitables.push_back(wall3);

        cudaDeviceSynchronize();
        hittable** list = new hittable * [hitables.size()];
        for (int i = 0; i < hitables.size(); i++) {
            list[i] = hitables[i];
            list[i]->create_hittable_on_gpu();
        }
        cudaDeviceSynchronize();
        world = new hittable_list(list, hitables.size());
        world->create_hittable_on_gpu();
        world->copy_list_to_gpu();

        background_color = color(0, 0, 0);
        point3 lookfrom = point3(278, 278, -800);
        point3 lookat = point3(278, 278, 0);
        float vfov = 40.0f;
        auto dist_to_focus = 10.0;
        auto aperture = 0.1;
        cam = new camera(lookfrom, lookat, vec3(0, 1, 0), vfov, 1.0f, aperture, dist_to_focus);
        cam->create_camera_on_gpu();

        cudaDeviceSynchronize();
    }

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

        auto material1 = new lambertian(color(0.4f, 0.2f, 0.1f));
        material1->create_material_on_gpu();
        auto sph = new sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, material1);
        hitables.push_back(sph);
        

        auto material2 = new lambertian(color(0.4f, 0.2f, 0.1f));
        material2->create_material_on_gpu();
        auto sph2 = new sphere(point3(-4, 1, 0), 1.0f, material2);
        hitables.push_back(sph2);

        auto material3 = new lambertian(color(0.7, 0.6, 0.5));
        material3->create_material_on_gpu();
        auto sph3 = new sphere(point3(4, 1, 0), 1.0f, material3);
        hitables.push_back(sph3);

        auto difflight = new diffuse_light(color(1, 1, 1));
        difflight->create_material_on_gpu();
        auto rect = new xy_rect(3, 5, 1, 3, -2, difflight);
        hitables.push_back(rect);
        

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

        background_color = color(0.0f, 0.0f, 0.0f);
    }

    scene(std::ifstream& file, float aspect_ratio) {
        std::string line;
        loadOptions(file, line);
        loadCamera(file, line, aspect_ratio);
        loadMaterials(file, line);
        loadObjects(file, line);

        for (int i = 0; i < materials.size(); i++) {
            materials[i]->create_material_on_gpu();
        }
        cudaDeviceSynchronize();

        hittable** list = new hittable * [hitables.size()];
        for (int i = 0; i < hitables.size(); i++) {
            list[i] = hitables[i];
            list[i]->create_hittable_on_gpu();
        }
        cudaDeviceSynchronize();
        world = new hittable_list(list, hitables.size());
        world->create_hittable_on_gpu();
        world->copy_list_to_gpu();
        cam->create_camera_on_gpu();
        cudaDeviceSynchronize();
    }

    void loadCamera(std::ifstream& file, std::string& line, float aspect_ratio) {
        point3 lookfrom;
        point3 lookat;
        vec3 vup = vec3(0, 1, 0);
        float vfov = 20;
        float dist_to_focus = 10;
        float aperture = 0.1f;

        do {
            std::getline(file, line);
            std::vector<std::string> maps;
            maps = get_key_value(line);
            if (maps[0] == "lookfrom") {
                lookfrom.load_from_string(maps[1]);
            }
            else if (maps[0] == "lookat") {
                lookat.load_from_string(maps[1]);
            }
            else if (maps[0] == "vfov") {
                vfov = std::stof(maps[1]);
            }
            else if (maps[0] == "distance_to_focus") {
                dist_to_focus = std::stof(maps[1]);
            }
            else if (maps[0] == "aperture") {
                aperture = std::stof(maps[1]);
            }
            else if (maps[0] == "vup") {
                vup.load_from_string(maps[1]);
            }

        } while (line != "Materials");
        cam = new camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus);
    }

    void loadOptions(std::ifstream& file, std::string& line) {
        do {
            std::getline(file, line);
            std::vector<std::string> maps;
            maps = get_key_value(line);
            if (maps[0] == "background_color") {
                color bc;
                bc.load_from_string(maps[1]);
                background_color = bc;
            }
          
        } while (line != "Camera");
    }

    void loadMaterials(std::ifstream& file, std::string& line) {
        do {
            std::getline(file, line);
            if (line == "Lambertian") {
                lambertian* mat = lambertian::load_from_file(file);
                materials.push_back(mat);
            }
            else if (line == "Dielectric") {
                dielectric* mat = dielectric::load_from_file(file);
                materials.push_back(mat);
            }
            else if (line == "Metal") {
                metal* mat = metal::load_from_file(file);
                materials.push_back(mat);
            }
            else if (line == "Diffuse_light") {
                diffuse_light* mat = diffuse_light::load_from_file(file);
                materials.push_back(mat);
            }

        } while (line != "Objects");
    }

    void loadObjects(std::ifstream& file, std::string& line) {
        do {
            std::getline(file, line);
            if (line == "Sphere") {
                sphere* obj = sphere::init_from_file(file, materials);
                hitables.push_back(obj);
            }
            else if (line == "Cylinder") {
                cylinder2* obj = cylinder2::init_from_file(file, materials);
                hitables.push_back(obj);
            }
            else if (line == "Cone") {
                cone* obj = cone::init_from_file(file, materials);
                hitables.push_back(obj);
            }
            else if (line == "Xy_rect") {
                xy_rect* obj = xy_rect::load_from_file(file, materials);
                hitables.push_back(obj);
            }
            else if (line == "Xz_rect") {
                xz_rect* obj = xz_rect::load_from_file(file, materials);
                hitables.push_back(obj);
            }
            else if (line == "Yz_rect") {
                yz_rect* obj = yz_rect::load_from_file(file, materials);
                hitables.push_back(obj);
            }

        } while (line != "END");
    }

    ~scene() {
        for (int i = 0; i < world->list_size; i++) {
            delete hitables[i];
        }
        for (int i = 0; i < materials.size(); i++) {
            delete materials[i];
        }
        delete world->list;
        delete world;
        delete cam;

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
    }



    public :
        std::vector<hittable*> hitables;
        std::vector<material*> materials;
        hittable_list* world;
        color background_color; 
        camera* cam;
};