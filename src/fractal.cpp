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
        if (a.magnitude2() > 1000)
            return 1000;
    }
    return a.magnitude2();
}

void kernel_row(unsigned char* ptr) {
    #pragma omp parallel 
    {
        int num_threads = omp_get_num_threads();
        int curr_thread = omp_get_thread_num();
    for (int y = curr_thread; y < DIM; y += num_threads) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            double val = julia(x, y);

            if (val == 1000) { //diverged
                ptr[offset * 3 + 0] = 0;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            } else {
                double t = log(val + 1.0) / log(5.0);
                t*= 2.0;
                if (t > 1.0) t = 1.0;

                ptr[offset * 3 + 0] = (unsigned char)(255 * t);                  // R
                ptr[offset * 3 + 1] = (unsigned char)(100 * pow(t, 0.5));        // G
                ptr[offset * 3 + 2] = (unsigned char)(255 * pow(1 - t, 2));      // B   
            }
        }
    }
    }
}

void kernel_col(unsigned char* ptr) {
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int curr_thread = omp_get_thread_num();
    for (int y = 0; y < DIM; y++) {
            for (int x = curr_thread; x < DIM; x += num_threads) {  
            int offset = x + y * DIM;

            double val = julia(x, y);

            if (val == 1000) { //diverged
                ptr[offset * 3 + 0] = 0;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            } else {
                double t = log(val + 1.0) / log(5.0);
                t*= 2.0;
                if (t > 1.0) t = 1.0;

                ptr[offset * 3 + 0] = (unsigned char)(255 * t);                  // R
                ptr[offset * 3 + 1] = (unsigned char)(100 * pow(t, 0.5));        // G
                ptr[offset * 3 + 2] = (unsigned char)(255 * pow(1 - t, 2));      // B   
            }
        }
    }
    }
}

void kernel_rblk(unsigned char* ptr) {
    #pragma omp parallel 
    {
        int num_threads = omp_get_num_threads();
        int curr_thread = omp_get_thread_num();
        int block_size = (int) DIM/omp_get_max_threads();
        int end = curr_thread == num_threads - 1 ? DIM : block_size * (curr_thread + 1);
    for (int y = curr_thread*block_size; y < end; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            double val = julia(x, y);

            if (val == 1000) { //diverged
                ptr[offset * 3 + 0] = 0;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            } else {
                double t = log(val + 1.0) / log(5.0);
                t*= 2.0;
                if (t > 1.0) t = 1.0;

                ptr[offset * 3 + 0] = (unsigned char)(255 * t);                  // R
                ptr[offset * 3 + 1] = (unsigned char)(100 * pow(t, 0.5));        // G
                ptr[offset * 3 + 2] = (unsigned char)(255 * pow(1 - t, 2));      // B   
            }
        }
    }
    }
}

void kernel_cblk(unsigned char* ptr) {
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int curr_thread = omp_get_thread_num();
        int block_size = (int) DIM/omp_get_max_threads();
        int end = curr_thread == num_threads - 1 ? DIM : block_size * (curr_thread + 1);  
    for (int y = 0; y < DIM; y++) {
        for (int x = curr_thread*block_size; x < end; x++) {  
            int offset = x + y * DIM;

            double val = julia(x, y);

            if (val == 1000) { //diverged
                ptr[offset * 3 + 0] = 0;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            } else {
                double t = log(val + 1.0) / log(5.0);
                t*= 2.0;
                if (t > 1.0) t = 1.0;

                ptr[offset * 3 + 0] = (unsigned char)(255 * t);                  // R
                ptr[offset * 3 + 1] = (unsigned char)(100 * pow(t, 0.5));        // G
                ptr[offset * 3 + 2] = (unsigned char)(255 * pow(1 - t, 2));      // B   
            }
        }
    }
    }
}

void kernel_omp_for_static(unsigned char* ptr) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            double val = julia(x, y);

            if (val == 1000) { //diverged
                ptr[offset * 3 + 0] = 0;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            } else {
                double t = log(val + 1.0) / log(5.0);
                t*= 2.0;
                if (t > 1.0) t = 1.0;

                ptr[offset * 3 + 0] = (unsigned char)(255 * t);                  // R
                ptr[offset * 3 + 1] = (unsigned char)(100 * pow(t, 0.5));        // G
                ptr[offset * 3 + 2] = (unsigned char)(255 * pow(1 - t, 2));      // B   
            }
        }
    }
}

void kernel_omp_for_dynamic(unsigned char* ptr) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            double val = julia(x, y);

            if (val == 1000) { //diverged
                ptr[offset * 3 + 0] = 0;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            } else {
                double t = log(val + 1.0) / log(5.0);
                t*= 2.0;
                if (t > 1.0) t = 1.0;

                ptr[offset * 3 + 0] = (unsigned char)(255 * t);                  // R
                ptr[offset * 3 + 1] = (unsigned char)(100 * pow(t, 0.5));        // G
                ptr[offset * 3 + 2] = (unsigned char)(255 * pow(1 - t, 2));      // B   
            }
        }
    }
}

void kernel_serial(unsigned char* ptr) {
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            double val = julia(x, y);

            if (val == 1000) { //diverged
                ptr[offset * 3 + 0] = 0;
                ptr[offset * 3 + 1] = 0;
                ptr[offset * 3 + 2] = 0;
            } else {
                double t = log(val + 1.0) / log(5.0);
                t*= 2.0;
                if (t > 1.0) t = 1.0;

                ptr[offset * 3 + 0] = (unsigned char)(255 * t);                  // R
                ptr[offset * 3 + 1] = (unsigned char)(100 * pow(t, 0.5));        // G
                ptr[offset * 3 + 2] = (unsigned char)(255 * pow(1 - t, 2));      // B   
            }
        }
    }
}

/* Save image as PPM */
void save_ppm(const char* filename, unsigned char* data, int width, int height) {
    ofstream file(filename, ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<char*>(data), width * height * 3);
    file.close();
}

/* Executed a function and times it */
double timed_execute(unsigned char* ptr, std::function<void(unsigned char*)> func) {
    double start = omp_get_wtime();
    func(ptr);
    return omp_get_wtime() - start;
}

/* Output helper function */
void output(string func, double func_time, double s_time) {
    cout << func << ":\t" << func_time << "ms \t| Speedup: " << s_time/func_time << endl;
}

/* Runs multiple timed executions and returns the average time */
double timed_multirun(unsigned char* ptr, std::function<void(unsigned char*)> func, int runs) {
    double total_time = 0;
    for (int i = 0; i < runs; i++) {
        total_time += timed_execute(ptr, func);
    }
    return total_time/(double) runs;
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