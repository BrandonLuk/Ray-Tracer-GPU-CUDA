#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel.cuh"

#include "common/color.h"
#include "common/cuda_error_check.h"
#include "common/ray.h"
#include "common/vec3.h"

#include "shapes/sphere.h"
#include "shapes/sphere_cuda.cu"
#include "shapes/rectangle.h"
#include "shapes/rectangle_cuda.cu"

#include "float.h"

#include <stdio.h>

constexpr int MAX_REFLECTION_DEPTH = 2;                 // How many reflections per ray
constexpr float LIGHT_RAY_NUDGE = 0.005f;               // How much to nudge light ray origins so that they do not intersect with their own entities
constexpr float REFLECTION_INTENSITY_RETENTION = 0.8f;  // How much color/light is retained after every reflection bounce

// RayTrace_Kernel Occupancy
int block_size;
int min_grid_size;
int grid_size;

// Pointers to device side entity arrays
Sphere* d_spheres;
Rectangle* d_rectangles;
Light* d_lights;

__constant__ int d_spheres_cnt;
__constant__ int d_rectangles_cnt;
__constant__ int d_lights_cnt;


struct CollisionRecord {
    enum { SPHERE, RECTANGLE, NONE } collided_shape;
    int index;
};

/*
    Search through spheres and rectangles for an entity that the given ray intersects.
*/
__device__ CollisionRecord NearestIntersection(
    Sphere* spheres,
    Rectangle* rectangles,
    Ray& ray,
    Vec3& intersection)
{
    float distance;
    float temp_distance;
    Vec3 temp_intersection;

    CollisionRecord col{ CollisionRecord::NONE, 0 };
    distance = FLT_MAX; // Start with an impossibly large distance to compare with

    // Go through all entities in the scene
    for (int i = 0; i < d_spheres_cnt; ++i)
    {
        // Check if the ray intersects this sphere
        if (SphereRayIntersect(spheres[i], ray, temp_intersection))
        {
            // Check if this intersection is closer than one we have already found
            temp_distance = ray.origin.Distance(temp_intersection);

            // If so, mark it as such
            if (temp_distance < distance)
            {
                intersection = temp_intersection;
                distance = temp_distance;
                col.collided_shape = CollisionRecord::SPHERE;
                col.index = i;
            }
        }
    }

    for (int i = 0; i < d_rectangles_cnt; ++i)
    {
        // Check if the ray intersects this rectangle
        if (RectangleRayIntersect(rectangles[i], ray, temp_intersection))
        {
            // Check if this intersection is closer than one we have already found
            temp_distance = ray.origin.Distance(temp_intersection);

            // If so, mark it as such
            if (temp_distance < distance)
            {
                intersection = temp_intersection;
                distance = temp_distance;
                col.collided_shape = CollisionRecord::RECTANGLE;
                col.index = i;
            }
        }
    }


    return col;
}

/*
    Find the lighting value at a point.
*/
__device__ float RayLight(
    Sphere* spheres,
    Rectangle* rectangles,
    Light* lights,
    Vec3 incidence,
    Ray normal_at_incident)
{
    float light_additive = 0.0f;

    Ray light_ray;
    Vec3 intersection;
    float surface_angle_from_light;

    // Slightly exaggerate the origin of the light ray, so that it does not intersect with the entity we are trying to find the light value for.
    light_ray.origin = incidence + (normal_at_incident.direction * LIGHT_RAY_NUDGE);

    // Check each light to see if there is an uninterrupted path between it and the entity
    for (int i = 0; i < d_lights_cnt; ++i)
    {
        light_ray.direction = lights[i].origin - incidence;
        light_ray.direction = light_ray.direction.Normalize();


        // If the light_ray intersects some entity
        CollisionRecord rec = NearestIntersection(spheres, rectangles, light_ray, intersection);
        if (rec.collided_shape != CollisionRecord::NONE && light_ray.origin.Distance(intersection) < light_ray.origin.Distance(lights[i].origin))
            continue;
        surface_angle_from_light = light_ray.direction.DotProduct(normal_at_incident.direction);
        if (surface_angle_from_light > 0.0f)
            light_additive += lights[i].intensity * surface_angle_from_light;
    }

    return light_additive;
}

/*
    Find the Color value associated with the given ray.
*/
__device__
Color RayColor(
    const Color& bg_color,
    Sphere* spheres,
    Rectangle* rectangles,
    Light* lights,
    Ray r)
{
    Color c(bg_color);
    Vec3 intersection;
    Ray normal_at_intersection;
    CollisionRecord rec;

    float light_retention = 1.0f;

    for (int i = 0; i < MAX_REFLECTION_DEPTH; ++i)
    {
        rec = NearestIntersection(spheres, rectangles, r, intersection);

        if (rec.collided_shape == CollisionRecord::SPHERE)
        {
            // If this sphere is not reflective or if we have reached the max reflection depth then just return this sphere's color
            if (!spheres[rec.index].reflective || i == MAX_REFLECTION_DEPTH - 1)
            {
                normal_at_intersection = SphereNormalAtPoint(spheres[rec.index], intersection);
                return light_retention * spheres[rec.index].color * RayLight(spheres, rectangles, lights, intersection, normal_at_intersection);
            }
            // Else we have hit a reflective sphere and are not at the max reflection depth, so reflect the incoming ray and make another pass
            else
            {
                r = SphereReflectedRay(spheres[rec.index], r, intersection);
                light_retention *= REFLECTION_INTENSITY_RETENTION;
            }
        }
        else if (rec.collided_shape == CollisionRecord::RECTANGLE)
        {
            if (!rectangles[rec.index].reflective || i == MAX_REFLECTION_DEPTH - 1)
            {
                normal_at_intersection = RectangleNormalAtPoint(rectangles[rec.index], intersection);
                return light_retention * rectangles[rec.index].color * RayLight(spheres, rectangles, lights, intersection, normal_at_intersection);
            }
            else
            {
                r = RectangleReflectedRay(rectangles[rec.index], r, intersection);
                light_retention *= REFLECTION_INTENSITY_RETENTION;
            }
        }
        else
        {
            return c * light_retention;
        }
    }

    return c;
}

__global__
void RayTrace_Kernel(
    Camera* camera,
    Color bg_color,
    Sphere* spheres,
    Rectangle* rectangles,
    Light* lights,
    Vec3 top_left_pixel_center,
    Vec3 q_x,
    Vec3 q_y,
    uchar4* des)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    unsigned int x, y;

    Vec3 point;
    Ray ray;

    int num_pixels = camera->frame_width * camera->frame_height;
    for (int i = index; i < num_pixels; i += stride)
    {


        // Convert i, which represents the "flat index" of the pixel array into 2-D coordinates.
        x = i % camera->frame_width;
        y = i / camera->frame_width;


        point = camera->origin + top_left_pixel_center + (q_x * x) - (q_y * y);

        ray.origin = camera->origin;
        ray.direction = point - camera->origin;
        ray.direction = ray.direction.Normalize();

        des[i] = RayColor(bg_color, spheres, rectangles, lights, ray);
    }
}

void RayTrace(
    Camera* camera,
    Color bg_color,
    uchar4* des)
{
    Vec3 viewport_center = camera->origin + ((float)camera->offset * camera->normal.Normalize());    // Center of our viewpower, given the current viewframe
    Vec3 target_vec = viewport_center - camera->origin; // Direction of the ray pointing from the camera to the viewport center
    Vec3 b = camera->roll_component.CrossProduct(target_vec);   // Vector represeting the positive x direction when facing the viewport center from the camera
    Vec3 target_vec_normal = target_vec.Normalize();
    Vec3 b_normal = b.Normalize();
    Vec3 v_normal = target_vec_normal.CrossProduct(b_normal);   // Vector represeting the positive y direction when facing the viewport center from the camera

    float g_x = camera->offset * tan(camera->FOV / 2.0);                 // Half the size of the viewport's width
    float g_y = g_x * camera->frame_height / (float)camera->frame_width;    // Half the size of the viewport's height

    Vec3 q_x = (2 * g_x / (float)(camera->frame_width - 1)) * b_normal;   // Pixel shifting vector along the width
    Vec3 q_y = (2 * g_y / (float)(camera->frame_height - 1)) * v_normal;  // Pixel shifting vector along the height

    Vec3 top_left_pixel_center = (target_vec_normal * camera->offset) - (g_x * b_normal) + (g_y * v_normal);

    RayTrace_Kernel << <grid_size, block_size >> > (
        camera,
        bg_color,
        d_spheres,
        d_rectangles,
        d_lights,
        top_left_pixel_center,
        q_x,
        q_y,
        des
        );

    cudaErrorChk(cudaPeekAtLastError());
    cudaErrorChk(cudaDeviceSynchronize());
}

void CalcKernelBlockSize(Camera* camera)
{
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        (void*)RayTrace_Kernel,
        0,
        camera->frame_width * camera->frame_height
    );
    grid_size = (camera->frame_width * camera->frame_height + block_size - 1) / block_size;
}

void CpyEntitiesToDevice(
    Sphere* spheres,
    int spheres_cnt,
    Rectangle* rectangles,
    int rectangles_cnt,
    Light* lights,
    int lights_cnt)
{
    cudaMalloc(&d_spheres, spheres_cnt * sizeof(Sphere));
    cudaMalloc(&d_rectangles, rectangles_cnt * sizeof(Rectangle));
    cudaMalloc(&d_lights, lights_cnt * sizeof(Light));
    cudaMemcpy(d_spheres, spheres, spheres_cnt * sizeof(Sphere), cudaMemcpyDefault);
    cudaMemcpy(d_rectangles, rectangles, rectangles_cnt * sizeof(Rectangle), cudaMemcpyDefault);
    cudaMemcpy(d_lights, lights, lights_cnt * sizeof(Light), cudaMemcpyDefault);

    cudaMemcpyToSymbol(d_spheres_cnt, &spheres_cnt, sizeof(spheres_cnt));
    cudaMemcpyToSymbol(d_rectangles_cnt, &rectangles_cnt, sizeof(rectangles_cnt));
    cudaMemcpyToSymbol(d_lights_cnt, &lights_cnt, sizeof(lights_cnt));
}

void CleanUp()
{
    cudaErrorChk(cudaFree(d_spheres));
    cudaErrorChk(cudaFree(d_rectangles));
    cudaErrorChk(cudaFree(d_lights));
}