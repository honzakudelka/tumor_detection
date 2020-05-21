#include <algorithm>
#include <cstring>
#include <iostream>
#include "cpp_slide_paraboloid.h"

#include <future>
#include <thread>
#include <chrono>

using namespace std;

enum filter_direction { X_DIR = 1, Y_DIR, DIAG_1A, DIAG_1B, DIAG_2A, DIAG_2B };

struct Direction {
    unsigned int start_line;
    unsigned int n_lines;
    unsigned int line_inc;
    unsigned int point_inc;
    int length;

    Direction() : start_line(0), n_lines(0), line_inc(0), point_inc(0), length(-1) {}
};

Direction get_line_properties( unsigned int dir, unsigned int w, unsigned int h)
{
    Direction output;

    switch (dir){
        case X_DIR:
            output.n_lines = h;
            output.line_inc = w;
            output.point_inc = 1;
            output.length = w;
            break;

        case Y_DIR:
            output.n_lines = w;
            output.line_inc = 1;
            output.point_inc = w;
            output.length = h;
            break;

        case DIAG_1A:
            output.n_lines = w - 2;
            output.line_inc = 1;
            output.point_inc = w + 1;
            break;

        case DIAG_1B:
            output.start_line = 1;
            output.n_lines = h - 2;
            output.line_inc = w;
            output.point_inc = w + 1;
            break;

        case DIAG_2A:
            output.start_line = 2;
            output.n_lines = w;
            output.line_inc = 1;
            output.point_inc = w - 1;
            break;

        case DIAG_2B:
            output.n_lines = h - 2;
            output.line_inc = w;
            output.point_inc = w - 1;
            break;
    }

    return output;

}

unsigned int get_length(unsigned int dir, unsigned int w, unsigned int h, unsigned int i)
{
    unsigned int output = 0;
    switch (dir){
        case DIAG_1A:
            output = min(h, w-i);
            break;
        case DIAG_1B:
            output = min(h-i, w);
            break;
        case DIAG_2A:
            output = min(h, i+1);
            break;
        case DIAG_2B:
            output = min(h - i, w);
            break;
    }
    return output;
}

float* line_slide_parabola(float* pixels, unsigned int px_length, unsigned int start_px, unsigned int px_inc, unsigned int length,
float coeff2, float* cache, unsigned int * next_point, float* corrected_edges = nullptr)
{
    float min_v = 1.e5;
    float v_prev1 = 0.f;
    float v_prev2 = 0.f;
    float curvature_test = 1.999f * coeff2;
    unsigned int last_point = 0;

    unsigned int first_corner = length - 1;
    unsigned int last_corner = 0;

    unsigned int p = start_px;
    for( unsigned int i=0; i < length; i++, p+=px_inc)
    {
        const float v = fmin(fmax(pixels[p], -255.0f), 255.0f);
        cache[i] = v;
        if (v < min_v) min_v = v;
        if (i >= 2 && (2 * v_prev1 - v_prev2 - v < curvature_test))
        {
            next_point[last_point] = i - 1;
            last_point = i - 1;
        }

        v_prev2 = v_prev1;
        v_prev1 = v;

        if (p < 0 || p > px_length - 1) break;
    }
    next_point[last_point] = length - 1;
    next_point[length-1] = 1e6;

    unsigned int i1 = 0;
    while (i1 < length - 1)
    {
        auto v1 = cache[i1];
        auto search_end = length;

        auto j = next_point[i1];
        int recalculate_limit = 0;
        float min_slope = 1.e5;
        unsigned int i2 = 0;
        while (j < search_end)
        {

            auto v2 = cache[j];
            float slope = (v2 - v1) / (j - i1) + coeff2 * (j - i1);

            if (slope < min_slope)
            {
                min_slope = slope;
                i2 = j;
                recalculate_limit = -3;
            }

            if (recalculate_limit == 0)
            {
                auto b = 0.5 * min_slope / coeff2;
                unsigned int max_search = i1 + static_cast<unsigned int>(b + sqrt(b * b + (v1 - min_v) / coeff2) + 1);
                if (0 < max_search && max_search < search_end) search_end = max_search;
            }

            j = next_point[j];
            recalculate_limit++;
        }

        if (i1 == 0) first_corner = i2;
        if (i2 == length - 1) last_corner = i1;

        for (unsigned int jj = i1 + 1; jj < i2; jj++)
        {
            unsigned int px = start_px + jj * px_inc;

            if (px < 0 || px > px_length - 1)
                break;

            const float a = jj - i1;
            pixels[px] = v1 + a * (min_slope - a * coeff2);
        }
        i1 = i2;

    }

/*    if corrected_edges is not None:

        if 4 * first_corner >= length:
            last_corner = length - 1
        if 4 * (length - 1 - last_corner) >= length:
            last_corner = length - 1

        v1 = cache[first_corner]
        v2 = cache[last_corner]
        slope = (v2 - v1) / (last_corner - first_corner)
        value0 = v1 - slope * first_corner
        coeff6 = 0
        mid = 0.5 * (last_corner + first_corner)
        for ii in range((length + 2) // 3, (2 * length) // 3 + 1):
            dx = (i - mid) * 2. / (last_corner - first_corner)
            poly6 = dx * dx * dx * dx * dx * dx - 1
            if cache[ii] < value0 + slope * ii + coeff6 * poly6:
                coeff6 = -(value0 + slope * i - cache[i]) / poly6

        dx = (first_corner - mid) * 2. / (last_corner - first_corner)
        corrected_edges[0] = value0 + coeff6 * (dx * dx * dx * dx * dx * dx - 1.) + \
                             coeff2 * first_corner * first_corner
        dx = (last_corner - mid) * 2. / (last_corner - first_corner)
        corrected_edges[1] = value0 + (length - 1) * slope + coeff6 * (dx * dx * dx * dx * dx * dx - 1.) + \
                             coeff2 * (length - 1 - last_corner) * (length - 1 - last_corner)
*/
    return corrected_edges;
}

void _filter1d(float* pixels, unsigned int px_length, unsigned int w, unsigned int h, unsigned int dir,
float coeff2, float* cache, unsigned int *next_point)
{
    auto line_dir = get_line_properties(dir, w, h);

    for (unsigned int i=line_dir.start_line; i < line_dir.n_lines; i++)
    {
        unsigned int start_px = i * line_dir.line_inc;
        if (dir == DIAG_2B)
            start_px += w - 1;

        // dynamic estimation of length for diagonal elements
        if (line_dir.length < 0)
            line_dir.length = get_length(dir, w, h, i);

        line_slide_parabola(pixels, px_length, start_px, line_dir.point_inc,
                            line_dir.length, coeff2, cache, next_point, nullptr);
    }

//    std::cout << "At end of filter1D with direction " << dir << "\n";
//    for (unsigned int i=5120; i<5140; i++)
//    {
//        std::cout << pixels[i] << " ";
//    }
//    std::cout << std::endl;

}

void sliding_paraboloid_float_background(float* pixels, int wi, int he, float radius,
                                         bool pre_smooth, bool correct)
{

    unsigned int w = static_cast<unsigned int>(wi);
    unsigned int h = static_cast<unsigned int>(he);

    unsigned int array_len = max(w, h);
    float *cache = new float[array_len];
    unsigned int *next_point = new unsigned int[array_len];

    unsigned int px_length = w * h;

    //std::cout << "Calling sliding paraboloid with parameters \n" <<
    //"width: " << wi << ", height: " << he << ", radius: " << radius << std::endl;

    for (unsigned int i=0; i<array_len; i++)
    {
        cache[i] = 0.f;
        next_point[i] = 0u;
    }

    float coeff2 = 0.5 / radius;
    float coeff2diag = 1. / radius;

    /*if (correct)
        // correct corners
        _correct_corners(pixels, w, h, coeff2, cache, next_point);*/

    for (unsigned int i=5120; i<5140; i++)
    {
        std::cout << pixels[i] << " ";
    }
    std::cout << std::endl;


    try
    {
        _filter1d(pixels, px_length, w, h, X_DIR, coeff2, cache, next_point);
        _filter1d(pixels, px_length, w, h, Y_DIR, coeff2, cache, next_point);
        _filter1d(pixels, px_length, w, h, X_DIR, coeff2, cache, next_point);

        std::chrono::system_clock::time_point timeout_passed
            = std::chrono::system_clock::now() + std::chrono::seconds(3);

        std::promise<int> p1;
        std::future<int> f_completes = p1.get_future();
        std::thread([](std::promise<int> p1, float* pixels, uint px_length, uint w, uint h, float coeff2diag, float* cache, unsigned int* next_point){
                _filter1d(pixels, px_length, w, h, DIAG_1A, coeff2diag, cache, next_point);
                _filter1d(pixels, px_length, w, h, DIAG_1B, coeff2diag, cache, next_point);
                _filter1d(pixels, px_length, w, h, DIAG_2A, coeff2diag, cache, next_point);
                _filter1d(pixels, px_length, w, h, DIAG_2B, coeff2diag, cache, next_point);
                _filter1d(pixels, px_length, w, h, DIAG_1A, coeff2diag, cache, next_point);
        },
        std::move(p1), pixels, px_length, w, h, coeff2diag, cache, next_point).detach();

        if (std::future_status::ready != f_completes.wait_until(timeout_passed))
        {
            std::cerr << "Skipped diagonal directions due to timeout";
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught an exception during filter1D: \n" << e.what() << std::endl;
    }

    delete [] cache;
    delete [] next_point;

}
