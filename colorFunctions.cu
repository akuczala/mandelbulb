//Color functions needed for shading and coloring

//integer clamp function
__device__ __host__ int clamp(int value, int lower, int upper)
{
  if(value < lower)
    return lower;
  if(value > upper)
    return upper;
  return value;
}
//convert RGB values to int #RRGGBB
__device__ __host__ int color(int r, int g, int b)
{
  //bound r,g,b values
  r = clamp(r,0,255);
  g = clamp(g,0,255);
  b = clamp(b,0,255);
  return (r << 16) + (g << 8) + b; // #RRGGBB
}

//color component methods
__device__ __host__ int getr(int col) //get red
{
  return (col >> 16) & 0xFF;
}
__device__ __host__ int getg(int col) //green
{
  return (col >> 8) & 0xFF;
}
__device__ __host__ int getb(int col) //blue
{
  return col & 0xFF;
}

//color operation methods

//multiply R,G,B by number (shading)
__device__ int colorShade(int col, float shade)
{
  return color(int(getr(col)*shade),
    int(getg(col)*shade),
    int(getb(col)*shade));
}
//average respective R, G, B components of two colors
__device__ int averageColors(int col1, int col2)
{
  return color((int)(getr(col1)/2.+getr(col2)/2.),
    int(getg(col1)/2.+getg(col2)/2.),
    int(getb(col1)/2.+getb(col2)/2.));
}
//weighted average
__device__ int averageColors(int col1, int col2,float weight)
{
  return color((int)(getr(col1)*weight+getr(col2)*(1-weight)),
    int(getg(col1)*weight+getg(col2)*(1-weight)),
    int(getb(col1)*weight+getb(col2)*(1-weight)));
}
//sum respective R,G,B components of 2 colors
__device__ int addColors(int col1, int col2)
{
  return color((int)(getr(col1)+getr(col2)),
    (int)(getg(col1)+getg(col2)),
    (int)(getb(col1)+getb(col2)));
}