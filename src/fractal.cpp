#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include <functional>
#include <cmath>
#include <png.h>
using namespace std;

//#define DIM 7680
#define DIM 12288

struct cuComplex {
    float r;
    float i;
    cuComplex(float a, float b) : r(a), i(b) {}
    float magnitude2(void) { return r * r + i * i; }
    cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

double julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.7269, 0.1889);
    cuComplex a(jx, jy);

    for (int i = 0; i < 300; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000) {
            double smooth = i + 1.0 - log2(log2(sqrt(a.magnitude2())));
            return smooth;
        }
    }
    return -1; //inside set
}

void kernel_omp_for_dynamic(unsigned char* ptr) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            double val = julia(x, y);

            if (val < 0) { //inside set
                ptr[offset * 3 + 0] = 0;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            } else {
                double t = val / 300;

                // Power curve — pushes mid-tones darker, concentrates colour near boundary
                t = pow(t, 0.4);

                // Red/purple -> blue -> black
                double r = 0.5 + 0.5 * cos(2.0 * M_PI * (t * 1.0 + 0.0));
                double g = 0.5 + 0.5 * cos(2.0 * M_PI * (t * 1.0 + 0.45));
                double b = 0.5 + 0.5 * cos(2.0 * M_PI * (t * 1.0 + 0.35));

                ptr[offset * 3 + 0] = (unsigned char)(255 * r);
                ptr[offset * 3 + 1] = (unsigned char)(255 * g);
                ptr[offset * 3 + 2] = (unsigned char)(255 * b);
            }
        }
    }
}

bool save_png(const char* filename, unsigned char* data, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        cerr << "Failed to open file: " << filename << endl;
        return false;
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); return false; }
 
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); fclose(fp); return false; }
 
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }
 
    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8,
                 PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
 
    // Use compression level 1 — fast write, still much smaller than PPM
    png_set_compression_level(png, 1);
 
    png_write_info(png, info);
 
    // Write row by row
    for (int y = 0; y < height; y++) {
        png_write_row(png, data + y * width * 3);
    }
 
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return true;
}

int main(void) {
    cout << "Allocating " << (DIM * DIM * 3) / (1024 * 1024) << " MB for " 
         << DIM << "x" << DIM << " image..." << endl;
 
    unsigned char* image = new unsigned char[DIM * DIM * 3];
 
    cout << "Rendering with 24 threads (dynamic scheduling)..." << endl;
    double start = omp_get_wtime();
    kernel_omp_for_dynamic(image);
    double elapsed = omp_get_wtime() - start;
 
    cout << "Render time: " << elapsed << "s" << endl;
    cout << "Saving PNG..." << endl;
 
    if (save_png("fractal.png", image, DIM, DIM)) {
        cout << "Saved to fractal.png" << endl;
    } else {
        cerr << "PNG save failed." << endl;
    }
 
    delete[] image;
    return 0;
}