#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include <functional>
using namespace std;

#define DIM 768

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

int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    cuComplex c(-0.7269, 0.1889);
    cuComplex a(jx, jy);

    for (int i = 0; i < 300; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

void kernel_row(unsigned char* ptr) {
    #pragma omp parallel for schedule(static, 1)
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 3 + 0] = 255 * juliaValue; // R
            ptr[offset * 3 + 1] = 0;               // G
            ptr[offset * 3 + 2] = 0;               // B
        }
    }
}

void kernel_col(unsigned char* ptr) {
    #pragma omp parallel for schedule(static, 1)
    for (int x = 0; x < DIM; x++) {    
        for (int y = 0; y < DIM; y++) {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 3 + 0] = 255 * juliaValue; // R
            ptr[offset * 3 + 1] = 0;               // G
            ptr[offset * 3 + 2] = 0;               // B
        }
    }
}

void kernel_rblk(unsigned char* ptr) {
    int block_size = (int) DIM/omp_get_max_threads();
    #pragma omp parallel for schedule(static, block_size)
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 3 + 0] = 255 * juliaValue; // R
            ptr[offset * 3 + 1] = 0;               // G
            ptr[offset * 3 + 2] = 0;               // B
        }
    }
}

void kernel_cblk(unsigned char* ptr) {
    int block_size = (int) DIM/omp_get_max_threads();
    #pragma omp parallel for schedule(static, block_size)
    for (int x = 0; x < DIM; x++) {    
        for (int y = 0; y < DIM; y++) {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 3 + 0] = 255 * juliaValue; // R
            ptr[offset * 3 + 1] = 0;               // G
            ptr[offset * 3 + 2] = 0;               // B
        }
    }
}

void kernel_omp_for(unsigned char* ptr) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 3 + 0] = 255 * juliaValue; // R
            ptr[offset * 3 + 1] = 0;               // G
            ptr[offset * 3 + 2] = 0;               // B
        }
    }
}

void kernel_serial(unsigned char* ptr) {
    for (int y = 0; y < DIM; y++) {
        for (int x = 0; x < DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 3 + 0] = 255 * juliaValue;
            ptr[offset * 3 + 1] = 0;
            ptr[offset * 3 + 2] = 0;
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

double timed_execute(unsigned char* ptr, std::function<void(unsigned char*)> func) {
    double start = omp_get_wtime();
    func(ptr);
    return omp_get_wtime() - start;
}

void output(string func, double func_time, double s_time) {
    cout << func << ": " << func_time << "ms | Speedup: " << s_time/func_time << endl;
}

double timed_multirun(unsigned char* ptr, std::function<void(unsigned char*)> func, int runs) {
    double total_time = 0;
    for (int i = 0; i < runs; i++) {
        total_time += timed_execute(ptr, func);
    }
    return total_time/(double) runs;
}

int main(void) {
    unsigned char* image_s = new unsigned char[DIM * DIM * 3];
    unsigned char* image_p = new unsigned char[DIM * DIM * 3];

    double time_s, time_r, time_c, time_rblk, time_cblk, time_p;

    int runs = 5;

    /* Serial run */
    time_s = timed_multirun(image_s, kernel_serial, runs);

    /* 1D Rowwise */
    time_r = timed_multirun(image_p, kernel_row, runs);

    /* 1D Colwise */
    time_c = timed_multirun(image_p, kernel_col, runs);

    /* 2D Rowblockwise */
    time_rblk = timed_multirun(image_p, kernel_rblk, runs);

    /* 2D Colblockwise */
    time_cblk = timed_multirun(image_p, kernel_cblk, runs);

    /* Parallel */
    time_p = timed_multirun(image_p, kernel_omp_for, runs);

    cout << "Elapsed time:\n";
    cout << "Serial time: " << time_s << "ms" << endl;
    output("1D Rowwise", time_r, time_s);
    output("1D Colwise", time_c, time_s);
    output("2D Row-block", time_rblk, time_s);
    output("2D Col-block", time_cblk, time_s);
    output("Omp For", time_p, time_s);

    /* Save result */
    save_ppm("output/fractal_serial.ppm", image_s, DIM, DIM);
    save_ppm("output/fractal_par.ppm", image_p, DIM, DIM);  

    delete[] image_s;
    delete[] image_p;
    return 0;
}