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

struct CollisionRecord {
    enum{SPHERE, RECTANGLE, NONE} Shape;
    int index;
};

/*
    Search through spheres and rectangles for an entity that the given ray intersects.
*/
__device__ CollisionRecord NearestIntersection(
    Sphere* spheres,
    int spheres_cnt,
    Rectangle* rectangles,
    int rectangles_cnt,
    Ray& ray,
    Vec3& intersection)
{
    double distance;
    double temp_distance;
    Vec3 temp_intersection;

    CollisionRecord col{ col.NONE, 0 };
    distance = DBL_MAX; // Start with an impossibly large distance to compare with
    
    // Go through all entities in the scene
    for (int i = 0; i < spheres_cnt; ++i)
    {
        // If this entity is not flagged to be ignored, and the ray has intersected it
        if (SphereRayIntersect(spheres[i], ray, temp_intersection))
        {
            // Check if this intersection is closer than one we have already found
            temp_distance = ray.origin.Distance(temp_intersection);

            // If so, mark it as such
            if (temp_distance < distance)
            {
                intersection = temp_intersection;
                distance = temp_distance;
                col.Shape = col.SPHERE;
                col.index = i;
            }
        }
    }

    for (int i = 0; i < rectangles_cnt; ++i)
    {
        // If this entity is not flagged to be ignored, and the ray has intersected it
        if (RectangleRayIntersect(rectangles[i], ray, temp_intersection))
        {
            // Check if this intersection is closer than one we have already found
            temp_distance = ray.origin.Distance(temp_intersection);

            // If so, mark it as such
            if (temp_distance < distance)
            {
                intersection = temp_intersection;
                distance = temp_distance;
                col.Shape = col.RECTANGLE;
                col.index = i;
            }
        }
    }
    

    return col;
}

/*
    Find the lighting value at a point.
*/
__device__ double RayLight(
    Sphere* spheres,
    int spheres_cnt,
    Rectangle* rectangles,
    int rectangles_cnt,
    Light* lights,
    int lights_cnt,
    Vec3 incidence,
    Ray normal_at_incident)
{
    double light_additive = 0.0;

    Ray light_ray;
    Vec3 intersection;
    double surface_angle_from_light;

    // Slightly exaggerate the origin of the light ray, so that it does not intersect with the entity we are trying to find the light value for.
    light_ray.origin = incidence * 0.05;

    // Check each light to see if there is an uninterrupted path between it and the entity
    for (int i = 0; i < lights_cnt; ++i)
    {
        light_ray.direction = lights[i].origin - incidence;
        light_ray.direction = light_ray.direction.Normalize();


        // If the light_ray intersects some entity
        CollisionRecord rec = NearestIntersection(spheres, spheres_cnt, rectangles, rectangles_cnt, light_ray, intersection);
        if (rec.Shape != rec.NONE && light_ray.origin.Distance(intersection) < light_ray.origin.Distance(lights[i].origin))
            continue;
        surface_angle_from_light = light_ray.direction.DotProduct(normal_at_incident.direction);
        if (surface_angle_from_light > 0.0)
            light_additive += lights[i].intensity * surface_angle_from_light;
    }

    return light_additive;
}

/*
    Find the Color value associated with the given ray.
*/
__device__
Color RayColor(
    Color& bg_color,
    Sphere* spheres,
    int spheres_cnt,
    Rectangle* rectangles,
    int rectangles_cnt,
    Light* lights,
    int lights_cnt,
    Ray& r)
{
    Color c(bg_color);
    Vec3 intersection;
    Ray normal_at_intersection;
    CollisionRecord rec;

    // If an entity was intersected by the ray
    rec = NearestIntersection(spheres, spheres_cnt, rectangles, rectangles_cnt, r, intersection);
    if (rec.Shape != rec.NONE)
    {
        if (rec.Shape == rec.SPHERE)
        {
            c = spheres[rec.index].color;
            normal_at_intersection = SphereNormalAtPoint(spheres[rec.index], intersection);
        }
        else
        {
            c = rectangles[rec.index].color;
            normal_at_intersection = RectangleNormalAtPoint(rectangles[rec.index], intersection);
        }
    }

    //rather check to see if entity origin is above camra origin
    double light_additive = RayLight(spheres, spheres_cnt, rectangles, rectangles_cnt, lights, lights_cnt, intersection, normal_at_intersection);

    c = c * light_additive;


    return c;
}

__global__
void RayTrace_Kernel_Thread(
    Camera* camera,
    Color bg_color,
    Sphere* spheres,
    int spheres_cnt,
    Rectangle* rectangles,
    int rectangles_cnt,
    Light* lights,
    int lights_cnt,
    Vec3 top_left_pixel_center,
    Vec3 q_x,
    Vec3 q_y,
    uchar4* des)
{


    extern __shared__ Sphere shared_mem[];
    Sphere* d_spheres = shared_mem;
    Rectangle* d_rectangles = (Rectangle*)&d_spheres[spheres_cnt];

    if (threadIdx.x < spheres_cnt)
        d_spheres[threadIdx.x] = spheres[threadIdx.x];
    else if (threadIdx.x < spheres_cnt + rectangles_cnt)
        d_rectangles[threadIdx.x - spheres_cnt] = rectangles[threadIdx.x - spheres_cnt];
    __syncthreads();

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
        
        des[i] = RayColor(bg_color, d_spheres, spheres_cnt, d_rectangles, rectangles_cnt, lights, lights_cnt, ray);
    }
}

void RayTrace(
    Camera* camera,
    Color bg_color,
    Sphere* spheres,
    int spheres_cnt,
    Rectangle* rectangles,
    int rectangles_cnt,
    Light* lights,
    int lights_cnt,
    uchar4* des)
{
    int blockSize = 256;
    int numBlocks = ((camera->frame_width * camera->frame_height) + blockSize - 1) / blockSize;

    Vec3 viewport_center = camera->origin + (camera->offset * camera->normal.Normalize());    // Center of our viewpower, given the current viewframe
    Vec3 target_vec = viewport_center - camera->origin; // Direction of the ray pointing from the camera to the viewport center
    Vec3 b = camera->roll_component.CrossProduct(target_vec);   // Vector represeting the positive x direction when facing the viewport center from the camera
    Vec3 target_vec_normal = target_vec.Normalize();
    Vec3 b_normal = b.Normalize();
    Vec3 v_normal = target_vec_normal.CrossProduct(b_normal);   // Vector represeting the positive y direction when facing the viewport center from the camera

    double g_x = camera->offset * tan(camera->FOV / 2);                 // Half the size of the viewport's width
    double g_y = g_x * camera->frame_height / (double)camera->frame_width;    // Half the size of the viewport's height

    Vec3 q_x = (2 * g_x / (double)(camera->frame_width - 1)) * b_normal;   // Pixel shifting vector along the width
    Vec3 q_y = (2 * g_y / (double)(camera->frame_height - 1)) * v_normal;  // Pixel shifting vector along the height

    Vec3 top_left_pixel_center = (target_vec_normal * camera->offset) - (g_x * b_normal) + (g_y * v_normal);
 
    RayTrace_Kernel_Thread << <numBlocks, blockSize, (spheres_cnt * sizeof(Sphere) + rectangles_cnt * sizeof(Rectangle))>> > (
        camera,
        bg_color,
        spheres,
        spheres_cnt,
        rectangles,
        rectangles_cnt,
        lights,
        lights_cnt,
        top_left_pixel_center,
        q_x,
        q_y,
        des
        );

    cudaErrorChk(cudaPeekAtLastError());
    cudaErrorChk(cudaDeviceSynchronize());
}
