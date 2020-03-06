//3D vector class, floating point precision
#include <iostream>
#include <cmath>

using namespace std;

class Vec
{
public:
	float x,y,z;
	__device__ __host__ Vec()
	{
		x=0; y=0; z=0;
	}
	__device__ __host__ Vec(float x, float y, float z)
	{
		this->x = x; this->y = y; this->z = z;
	}
	__device__ __host__ Vec add(const Vec v)
	{
		return Vec(x + v.x,y + v.y,z + v.z);
	}
	void __device__ __host__ addTo(const Vec v) //increments vector
	{
		x += v.x; y += v.y; z += v.z;
	}
	__device__ __host__ Vec sub(const Vec v)
	{
		return Vec(x - v.x, y - v.y, z - v.z);
	}
	__device__ __host__ Vec times(float a)
	{
		return Vec(x*a,y*a,z*a);
	}
	__device__ __host__ float dot(const Vec u)
	{
		return x*u.x + y*u.y + z*u.z;
	}
	__device__ __host__ float mag()
	{
		return sqrt(squared());
	}
	__device__ __host__ Vec cross(const Vec b)
	{
	float outx = y*b.z - z*b.y;
	float outy = z*b.x - x*b.z;
	float outz = x*b.y - y*b.x;
	return Vec(outx,outy,outz);
	}
	__device__ __host__ float squared()
	{
	return x*x+y*y+z*z;
	}
	__device__ __host__ Vec copy()
	{
		return Vec(x,y,z);
	}
	__device__ __host__ Vec unit()
	{
	return times(1.0/mag());
	}
	void print()
	{
	cout << x << "," << y << "," << z << endl;
	}
};