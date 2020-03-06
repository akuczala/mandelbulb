//MANDELBULB WITH CUDA
//Alexander Kuczala 2015
//akuczala@ucsd.edu

//Input: base name of output files

//Outputs binary .rgb file of pixels encoded as 24 bit colors #RRGGBB
//can be converted to image file with program such as ImageMagick (`convert' in linux)

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sys/time.h>
#include "Vec.cu"
#include "colorFunctions.cu"

//number of runs
const int runs = 1;
//number of frames/run
int frames = 10;
//fractal parameters
const bool isJulia = false; //if true draws a julia set instead of mandelbulb
const float power = 8; //power of recursion function


//rendering parameters

const float specularity = 0.5; //intensity of specularity highlights
const float specularExponent = 3; //larger values -> smaller highlights
const float fogDistance = 4; //distance at which fog completely occludes objects


const float lightIntensity = 1;


const float cameraAngle = 1.0; //divergence of camera rays

//fractal calculation parameters
//default values: bailout = 6, maxIterations = 8
const float bailout = 12; //value of r to terminate at, lower -> smoother, less detailed
const int maxIterations = 32; //more iterations, more accurate fractal

//ray stepping parameters
const float epsilonScale = 0.1; //default value = 1
const float minEpsilon = 1E-7; //default = 1E-7
const int stepLimit = 5000; //number of allowed marching steps (default 100)
const float rayLengthLimit = 4; //maximum ray length allowed
         //(should be smaller or equal to than fog distance)

int* h_pixels; //host image array

//clip values to certain range
__device__ float clamp(float value, float lower, float upper)
{
	if(value < lower)
		return lower;
	if(value > upper)
		return upper;
	return value;
}

//Ray class performs operations of single Ray/processor
//All ray functions are performed on GPU
class Ray{

public:
  Vec dir; //Ray direction

  //Camera parameters
  Vec cameraPos; //camera position
  Vec cameraTarget; //camera target
  Vec cameraDir; //calculated direction of camera
  Vec cameraUp; //direction of camera's y axis
  Vec cameraRight; //direction of camera's x axis

  //Light position
  Vec lightPos;

  //const bool shadowsOn = false;
  
  //constant vector c for julia set
  Vec julia;

  //coloring variables
  int backgroundColor;

  float breakRadius; //keep track of breakout radius value for coloring
  float minDist; //keep track of minimum distant of orbits in recursion


  float eps; //intersection distance threshold
  float pixelScale; //ray stepping size
  int stepsTaken; //number of ray steps taken in last iteration

  int width, height; //image dimensions

  //Constructor
  __device__ Ray(int i, int j, Vec cameraPos, Vec cameraTarget, int width, int height)
  {
    //set width and height 
    this->width = width;
    this->height = height;

    pixelScale = 1.0/width; //scale of distance between rays

    //set camera parameters
    Vec cameraUp(0,0,1); //set direction of camera y axis
    this->cameraPos = cameraPos.copy(); 
    this->cameraTarget = cameraTarget.copy();
    this->cameraUp = cameraUp.copy();
    //set light position
    Vec lightPos(-2,-2,2);
    this->lightPos = lightPos;

    initCamera(); //set up orthogonal basis for camera
    dir = rayDir(i,j);

    //set julia constant
    Vec julia(0.8,-0.9,-0.4);

    //set background color
    backgroundColor = color(100,100,100);
  }
  //calculate ray direction from pixel address
__device__ Vec rayDir(int i, int j)
  {
    //scale the camera frame vectors to create the cone of rays
    float xscale = 1.0*(i-width/2.0)/width*cameraAngle;
    float yscale = -1.0*(j-height/2.0)/height*cameraAngle;

    Vec out = cameraDir.add(cameraRight.times(xscale)).add(cameraUp.times(yscale));
    return out.unit();
  }
  //Single ray marching step with intital vector zed0
__device__ float traceStep(Vec zed0)
  {
    Vec c(0,0,0); //initialize c vector
    //c is either a constant (for julia) or the starting point (mandelbulb)
    if(isJulia)
      c = julia;
    else
      c = zed0.copy();
    Vec zed = zed0.copy(); 
    
    //convert initial zed to spherical coordinates
    float r =  zed.mag();
    float th = atan2(zed.y,zed.x);
    float ph = asin(zed.z/r);
    
    float dr = 1; //initial value of r derivative
    
    minDist = -1; //initialize minimum distance

    float powR, powRsin;
    int n=0;
    //zed iterations
    for(n=0; n<= maxIterations; n++)
    {
      //compute scalar derivative approximation
      powR = pow(r,power - 1);
      dr = dr*powR*power + 1;
      //iterate zed (zed = zed^p + c)
      powR = pow(r,power);
      powRsin = sin(power*ph);
      zed.x = powR*powRsin*cos(power*th);
      zed.y = powR*powRsin*sin(power*th);
      zed.z = powR*cos(power*ph);
      zed.addTo(c);

      r = zed.mag(); //new radius
    
      if(minDist < 0 ^ r < minDist) minDist = r; //update min distance
      if(r > bailout) break; //stop iterating if r exceeds bailout value
      
      //calculate new angles
      th = atan2(zed.y, zed.x); 
      ph = acos(zed.z / r); 
    }
    //memoize for coloring
    breakRadius = r;

    //return distance estimation value
    return 0.5*r*log(r)/dr;
  }
  //approximate normal vector to fractal surface
__device__ Vec getNormal(Vec zed)
{
    eps = eps/2.0;
    //calculate small finite differences around zed
    Vec zedx1 = zed.add(Vec(eps,0,0));
    Vec zedx2 = zed.sub(Vec(eps,0,0));
    
    Vec zedy1 = zed.add(Vec(0,eps,0));
    Vec zedy2 = zed.sub(Vec(0,eps,0));
    
    Vec zedz1 = zed.add(Vec(0,0,eps));
    Vec zedz2 = zed.sub(Vec(0,0,eps));
    
    //calculate normal to surface
    float dx = traceStep(zedx1) - traceStep(zedx2);
    float dy = traceStep(zedy1) - traceStep(zedy2);
    float dz = traceStep(zedz1) - traceStep(zedz2);
    Vec normal = Vec(dx,dy,dz);
    normal = normal.unit();
    
    return normal;
}
//ray stepping algorithm
__device__ float trace(Vec p0, Vec dir)
{
    Vec zed0 = p0.copy(); //initial point
    float rayLength = 0;
    eps = minEpsilon; //initial intersection threshold
    int maxSteps = int(1.0*stepLimit/epsilonScale);
    
    float distance = 0;
    int i;
    for(i = 0;i<maxSteps;i++)
    {
      
      distance = traceStep(zed0); //calculate maximum distance to fractal
      //step ray forward
      zed0 = zed0.add(dir.times(epsilonScale*distance));
      rayLength += epsilonScale*distance;

      //if ray length exceeds limit, assume no intersection and stop
      if(rayLength > rayLengthLimit)
        return -1;
      //stop if within intersection threshold
      if(distance < eps) break;
      //update intersection threshold
      eps = max(minEpsilon,pixelScale*rayLength);
      //println("eps= " + eps);
    }
    stepsTaken = i; //record steps taken
    //assume intersection if number of steps is exhausted
    //this can cause artifacts if the stepLimit is too small
    return rayLength;
}
//various routines for coloring
__device__ int stepColoring()
{
    int scale = 20;
    float r = 1.0*(stepsTaken%scale)/scale;
    float g = 1.0*((stepsTaken+scale/3)%scale)/scale;
    float b = 1.0*((stepsTaken+2*scale/3)%scale)/scale;
    r = abs(r-0.5)*2;
    g = abs(g-0.5)*2;
    b = abs(b-0.5)*2;
    return color(int(r*255),int(g*255),int(b*255));
}
__device__ int minOrbitColoring()
  {
    float scale = 0.4;
    float r,g,b;
    float spam;
    r = modf((minDist)/scale,&spam);
    g = modf((minDist+scale/3)/scale,&spam);
    b = modf((minDist+2*scale/3)/scale,&spam);
    r = abs(r-0.5)*2;
    g = abs(g-0.5)*2;
    b = abs(b-0.5)*2;
    return color(int(r*255),int(g*255),int(b*255));
  }
//returns fractal color
__device__ int getCol()
{
    return minOrbitColoring();
}
//simulate ambient light by shading
//based on number of steps taken and minimum orbit distance
__device__ float ambientOcclusion()
{
    //const float aoStrength = 1;
    const float emphasis = 0.58; //default

    int maxSteps = int(stepLimit/ epsilonScale);

    float ao = 1.0 - minDist*minDist;
    if(ao < 0)
      ao = 0;
   if(ao > 1)
     ao = 1;
    ao = 1.0 - ao;
    ao = ao*(1-1.0*stepsTaken/((float)(maxSteps))*2*emphasis);
    return clamp(ao,0.0,1.0);
}
//apply fog based on distance to point
__device__ float fog(float distance)
{
	return clamp(distance/fogDistance,0.0,1.0);
}

__device__ int rayTraceFractal()
{
  //Vec dir = rayDir(i,j);
  Vec pos = cameraPos;
  
	float distance = trace(pos,dir); //find distance with ray marching

	if(distance < 0) //negative distance means no intersection
		return backgroundColor;
	//intersection point of ray with surface
	Vec intersect = pos.add(dir.times(distance));
	//normal to surface
	Vec normal = getNormal(intersect);
	//shading for surface

  //calculate unit vector pointing from light to object
  Vec lightDir = intersect.sub(lightPos);
  lightDir = lightDir.unit();

  //calculate cos of angle between light ray and normal to sphere and use for shade
  float normDotLight = -normal.dot(lightDir);
  float shade = 0;
  if(normDotLight < 0) //if dot product is - then no shading
    shade = 0;
  else
    shade = normDotLight*lightIntensity;
  shade = abs(shade);

  //phong specularity-----
  //reflected light vector
  Vec reflect = lightDir.times(-1).sub(normal.times(2*normDotLight));
  float reflDotRay = -reflect.dot(dir);
  float specular = 0;
  if(reflDotRay < 0)
    specular = specularity*pow(abs(reflDotRay),specularExponent);
  //base color is lambertian shading
  int out = colorShade(getCol(),shade);
  //apply specularity
  out = addColors(out,colorShade(color(255,255,255),specular));
  //apply ambient occulsion
  out = colorShade(out,ambientOcclusion());
  //check for shadows.
  //if(shadowsOn)
  //{
    //create shadow detecting ray pointing from object to light
    //place ray's origin slightly above intersection point
    //push above surface by normal*eps
    //Vec shadowPos = intersect.copy().add(normal.times(eps));
    //Vec shadowDir = lightDir.times(-1);
    //float dist = trace(pos,dir); //compute distance to fractal along ray to light
    //if ray intersects a surface between object and light, cast shadow
    //if(dist > 0 && dist*dist < intersect.sub(lightPos).squared())
    //{
    //  return 0;
    //}
  //}
  //add fog
  out = averageColors(backgroundColor,out,fog(distance));
  
  return out;
}

//calculate frame vectors for camera
__device__ void initCamera()
{
	//points from camera to target
	cameraDir = cameraTarget.sub(cameraPos).unit();
	//use Graham Schmidt to make up vector orthogonal to dir
	cameraUp = cameraUp.sub(cameraDir.times(cameraUp.dot(cameraDir)));
	cameraUp = cameraUp.unit();
	//calculate right pointing camera frame vector
	cameraRight = cameraDir.cross(cameraUp).unit();
}
};
//end ray object----------------------------

//Kernel. One ray per thread.
__global__ void draw(int* pixels,int* width, int* height, Vec* cameraPos, Vec* cameraTarget)
{
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int n = (*width) * (*height);
	
  if(index < n)
  {
    int i = index%(*width);
    int j = index/(*width);
    Ray ray(i,j,*cameraPos,*cameraTarget,*width,*height);
    pixels[index] = ray.rayTraceFractal();
  }
	//
}
//write pixel color values as binary #RRGGBB to output file
void write(string outFileName,int width, int height)
{
	ofstream outFile;
  //open file for writing in binary
	outFile.open(outFileName.c_str(),ios::out | ios::binary);
  if(!outFile.is_open())
  {
    cout << "couldn't write to " << outFileName << endl;
    return;
  }
  cout << "writing to " << outFileName << endl;

	for(int i=0;i<width*height;i++)
	{
    int p = h_pixels[i];
    //put the bits in the right order (Read from left to right)
    unsigned int unp = (unsigned int)(color(getb(p),getg(p),getr(p)));
		//outFile << h_pixels[i];
    outFile.write((char*) &unp,3); //colors are 3 bytes long
	}
	outFile.close();
}

int main(int argc, char* argv[])
{
  
  //timer parameters
  struct timeval t1, t2;
  struct timezone tz;
  //time data arrays
  double time[runs];
  float kernelTime[runs];

  cudaError_t err;
  //run loop. can vary image size etc
  for(int run = 0; run< runs; run++)
  {
    //start timer------
    gettimeofday(&t1, &tz);

    //int h_width = (run+1)*100; //variable width
    int h_width = 1200; //constant width
        cout << "width = " << h_width << endl;
    //image size on host and device
    int h_height = h_width;
    int* d_width;int* d_height;

    int n = h_width*h_height; //number of pixels
    size_t size = sizeof(int)*n;
    size_t vecSize = sizeof(Vec);
    //allocate pixel array on host
    h_pixels = (int*)malloc(size); 

    int* d_pixels; //pixel array on device

    //Camera position and target
    Vec h_cameraPos;
    Vec h_cameraTarget;
    Vec* d_cameraPos;
    Vec* d_cameraTarget;
    
    //allocate memory on device
    //allocate image size on device
    err = cudaMalloc((void **) &d_width, sizeof(int));
    if(err != cudaSuccess) cout << "can't allocate memory for width on device" << endl;
    err = cudaMalloc((void **) &d_height, sizeof(int));
    if(err != cudaSuccess) cout << "can't allocate memory for height on device" << endl;
    //allocate pixel array on device
    err = cudaMalloc((void **) &d_pixels, size);
    if(err != cudaSuccess) cout << "can't allocate memory for pixel array on device" << endl;
    //allocate camera position and target
    err = cudaMalloc((void **) &d_cameraPos, vecSize);
    if(err != cudaSuccess) cout << "can't allocate memory for cameraPos on device" << endl;
    err = cudaMalloc((void **) &d_cameraTarget, vecSize);
    if(err != cudaSuccess) cout << "can't allocate memory for cameraTarget on device" << endl;
    
    //run animation

    //set initial and final values of camera target and position
    Vec cameraTargetInit(0,0,0);
    Vec cameraTargetFinal(0.6025440273509881, -0.7549067847481121, 0.5049324975811623);
    Vec cameraPosInit(1,-2,1.5);
    Vec cameraPosFinal = cameraTargetFinal.copy();

    float dt = 1.0/frames;
    float t = 0;

    for(int frame = 0;frame < frames; frame++)
    {
      cout << "Frame " << frame << "/" << frames << endl;

      //move towards fractal at exponentially decaying rate
      float distFrac = exp(-8*t);
      h_cameraPos = cameraPosInit.times(distFrac).add(cameraPosFinal.times(1-distFrac));
      h_cameraTarget  = cameraTargetInit.times(distFrac).add(cameraTargetFinal.times(1-distFrac));
      
      //copy image size to device
      err = cudaMemcpy(d_width, &h_width, sizeof(int), cudaMemcpyHostToDevice);
      if(err != cudaSuccess) cout << "can't copy width to device" << endl;
      err =cudaMemcpy(d_height, &h_height, sizeof(int), cudaMemcpyHostToDevice);
      if(err != cudaSuccess) cout << "can't copy height to device" << endl;
      //copy camera data to device
      err = cudaMemcpy(d_cameraPos, &h_cameraPos, vecSize, cudaMemcpyHostToDevice);
      if(err != cudaSuccess) cout << "can't copy cameraPos to device" << endl;
      err =cudaMemcpy(d_cameraTarget, &h_cameraTarget, vecSize, cudaMemcpyHostToDevice);
      if(err != cudaSuccess) cout << "can't copy cameraTarget to device" << endl;

      //start CUDA timer
      cudaEvent_t start, stop;
      cudaEventCreate(&start); cudaEventCreate(&stop);
      cudaEventRecord(start,0); //start kernel timer

      //----launch kernel-----
      int threadsPerBlock = 256;
      int blocksPerGrid = (n + threadsPerBlock -1)/threadsPerBlock;

      cout << "launching " << blocksPerGrid << " blocks of ";
      cout << threadsPerBlock << " threads" << endl;

      draw<<<blocksPerGrid, threadsPerBlock>>>(d_pixels,d_width,d_height, d_cameraPos, d_cameraTarget);
      
      //stop CUDA timer
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&kernelTime[run],start,stop);
      cudaEventDestroy(start);cudaEventDestroy(stop);
      cout << "kernel time: " << kernelTime[run] << endl;
      //check for kernel error
      err = cudaGetLastError();
      if(err != cudaSuccess) cout << "kernel failed: " << cudaGetErrorString(err) << endl;
      
      //copy results to hosts
      err = cudaMemcpy(h_pixels, d_pixels, size,cudaMemcpyDeviceToHost);
      if(err != cudaSuccess) cout << "can't copy to host" << endl;

      //if program has output filename, output to file
    	if(argc == 2)
    	{
        stringstream ss;
        ss << argv[1] << "_" << setfill('0') << setw(3) << frame << ".rgb" ;
        //ss << argv[1] << "_" << setfill('0') << setw(4) << h_width << ".rgb" ;
        string fileName;
        ss >> fileName;
    		write(fileName,h_width,h_height);
    	}
      //increment t
      t += dt;
    }
    //Deallocate memory
    cudaFree(d_pixels);
    cudaFree(d_cameraTarget); cudaFree(d_cameraPos);

    //stop timer---
    gettimeofday(&t2, &tz);
    time[run] = (t2.tv_sec-t1.tv_sec) + 1e-6*(t2.tv_usec-t1.tv_usec);
      cout << "Run time: " << time[run] << endl;
  }
  //reset GPU
  err = cudaDeviceReset();
  if(err != cudaSuccess) cout << "Couldn't reset GPU" << endl;

  //print runtimes
  cout << "Run times" << endl;
  for(int i=0;i<runs;i++)
    cout << time[i] << endl;
  cout << "Kernel times" << endl;
  for(int i=0;i<runs;i++)
    cout << kernelTime[i] << endl;

	return 0;
}