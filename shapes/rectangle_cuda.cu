#include "cuda_runtime.h"

#include "rectangle.h"

#include "common/ray.h"
#include "common/vec3.h"

#include <cmath>

#ifndef EPSILON
#define EPSILON 1e-7
#endif


__device__ inline bool RectangleRayIntersect(const Rectangle& rectangle, const Ray& ray, Vec3& intersection)
{
    /*
    * First find the intersection between the ray and the plane the rectangle exists on.
    */
    double dividend = (rectangle.a - ray.origin).DotProduct(rectangle.normal);
    double divisor = ray.direction.DotProduct(rectangle.normal);

    // If the ray and plane are parallel
    if (fabs(divisor) < EPSILON)
    {
        return false;
    }

    double distance = dividend / divisor;

    // If the point of intersection is "behind" the ray
    if (distance < 0.0)
        return false;
    intersection = ray.origin + (distance * ray.direction);

    /*
    * Now check if the point of intersection exists within the bounds of this rectangle.
    */

    Vec3 AB = rectangle.a - rectangle.b;
    Vec3 AC = rectangle.a - rectangle.c;
    Vec3 AI = rectangle.a - intersection;

    double ABAB_dot = AB.DotProduct(AB);
    double ACAC_dot = AC.DotProduct(AC);
    double AIAB_dot = AI.DotProduct(AB);
    double AIAC_dot = AI.DotProduct(AC);

    return(AIAB_dot >= 0.0 &&
        AIAC_dot >= 0.0 &&
        AIAB_dot <= ABAB_dot &&
        AIAC_dot <= ACAC_dot);
}

__device__ inline Ray RectangleNormalAtPoint(const Rectangle& rectangle, const Vec3& point)
{
    return { point, rectangle.normal };
}

__device__ inline Ray RectangleReflectedRay(const Rectangle& rectangle, const Ray& incoming_ray, const Vec3& intersection)
{
    Vec3 reflected_ray_direction = incoming_ray.direction - 2 * (incoming_ray.direction.DotProduct(rectangle.normal)) * rectangle.normal;
    reflected_ray_direction = reflected_ray_direction.Normalize();
    return Ray{ intersection, reflected_ray_direction };
}