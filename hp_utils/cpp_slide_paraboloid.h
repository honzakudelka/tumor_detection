#define DLLEXPORT extern "C"

extern "C" {
void sliding_paraboloid_float_background(float* pixels, int wi,
                                         int he, float radius,
                                         bool pre_smooth, bool correct);
}