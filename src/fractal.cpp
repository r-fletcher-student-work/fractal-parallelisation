#include <iostream>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include <functional>
#include <cmath>
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
    #pragma omp parallel for schedule(static, 1)
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

void kernel_col(unsigned char* ptr) {
    #pragma omp parallel for schedule(static, 1)
    for (int x = 0; x < DIM; x++) {    
        for (int y = 0; y < DIM; y++) {
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

void kernel_rblk(unsigned char* ptr) {
    int block_size = (int) DIM/omp_get_max_threads();
    #pragma omp parallel for schedule(static, block_size)
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

void kernel_cblk(unsigned char* ptr) {
    int block_size = (int) DIM/omp_get_max_threads();
    #pragma omp parallel for schedule(static, block_size)
    for (int x = 0; x < DIM; x++) {    
        for (int y = 0; y < DIM; y++) {
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

int main(void) {
    unsigned char* image_s = new unsigned char[DIM * DIM * 3];
    unsigned char* image_r = new unsigned char[DIM * DIM * 3];
    unsigned char* image_c = new unsigned char[DIM * DIM * 3];
    unsigned char* image_rb = new unsigned char[DIM * DIM * 3];
    unsigned char* image_cb = new unsigned char[DIM * DIM * 3];
    unsigned char* image_f = new unsigned char[DIM * DIM * 3];

    double time_s, time_r, time_c, time_rblk, time_cblk, time_p_s, time_p_d;

    // csv file for timings
    ofstream csv("timings.csv");
    csv << "threads,serial,row,col,rowblk,colblk,for_static,for_dynamic\n";

    const int runs = 10;

    int t;
    for (int i = 0; i <= 16; i+=2) {
        if (i == 0) t = 1;
        else t = i;

        omp_set_num_threads(t);
        cout << "+---------------------+" << endl;
        cout << "|  Thread count = " << t << "  |" << endl;
        cout << "+---------------------+" << endl;

        /* Serial run */
        time_s = timed_multirun(image_s, kernel_serial, runs);

        /* 1D Rowwise */
        time_r = timed_multirun(image_r, kernel_row, runs);

        /* 1D Colwise */
        time_c = timed_multirun(image_c, kernel_col, runs);

        /* 2D Rowblockwise */
        time_rblk = timed_multirun(image_rb, kernel_rblk, runs);

        /* 2D Colblockwise */
        time_cblk = timed_multirun(image_cb, kernel_cblk, runs);

        /* OMP for static scheduling */
        time_p_s = timed_multirun(image_f, kernel_omp_for_static, runs);

        /* OMP for dynamic scheduling */
        time_p_d = timed_multirun(image_f, kernel_omp_for_dynamic, runs);

        // append csv
        csv << t << ","
            << time_s << ","
            << time_r << ","
            << time_c << ","
            << time_rblk << ","
            << time_cblk << ","
            << time_p_s << ","
            << time_p_d << "\n";

        cout << "Elapsed time:\n";
        cout << "Serial time:\t" << time_s << "ms" << endl;
        output("1D Rowwise", time_r, time_s);
        output("1D Colwise", time_c, time_s);
        output("2D Row-block", time_rblk, time_s);
        output("2D Col-block", time_cblk, time_s);
        output("For static", time_p_s, time_s);
        output("For dynamic", time_p_d, time_s);

    }

    csv.close();

    /* Save result */
    save_ppm("output/fractal_serial.ppm", image_s, DIM, DIM);
    save_ppm("output/fractal_row.ppm", image_r, DIM, DIM);  
    save_ppm("output/fractal_col.ppm", image_c, DIM, DIM);  
    save_ppm("output/fractal_rb.ppm", image_rb, DIM, DIM);  
    save_ppm("output/fractal_cb.ppm", image_cb, DIM, DIM);  
    save_ppm("output/fractal_for.ppm", image_f, DIM, DIM);  

    delete[] image_s;
    delete[] image_r;
    delete[] image_c;
    delete[] image_rb;
    delete[] image_cb;
    delete[] image_f;
    return 0;
}