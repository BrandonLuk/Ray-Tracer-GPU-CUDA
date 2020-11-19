#include "cuda_runtime.h"

#include "sphere.h"

#include "common/ray.h"
#include "common/vec3.h"

#include <cmath>

__device__ inline bool SphereRayIntersect(const Sphere& sphere, const Ray& ray, Vec3& intersection)
{
    double v = ray.direction.DotProduct(sphere.origin - ray.origin);
    double discriminant = (sphere.radius * sphere.radius) - ((sphere.origin - ray.origin).DotProduct(sphere.origin - ray.origin) - (v * v));

    // There are no intersections
    if (discriminant < 0.0)
        return false;
    else
    {
        double distance = sqrt(discriminant);
        double t_param = v - distance;

        // If the t_param is negative, the intersection is "behind" the ray. For our purpose we throw this out and return false
        if (t_param < 0.0)
            return false;

        // An intersection exists and it is "in front" of the ray
        intersection = ray.origin + ((t_param)*ray.direction);
        return true;
    }
}

__device__ inline Ray SphereNormalAtPoint(const Sphere& sphere, const Vec3& point)
{
    Ray normal;

    normal.origin = sphere.origin;
    normal.direction = (point - sphere.origin).Normalize();

    return normal;
}

__device__ inline Ray SphereReflectedRay(const Sphere& sphere, const Ray& incoming_ray, const Vec3& intersection)
{
    Ray reflected_ray;

    reflected_ray.origin = intersection;

    Ray normal = SphereNormalAtPoint(sphere, intersection);

    double theta = -1 * normal.direction.DotProduct(incoming_ray.direction);
    reflected_ray.direction = incoming_ray.direction + (2 * theta * normal.direction);
    reflected_ray.direction = reflected_ray.direction.Normalize();

    return reflected_ray;
}