// Simple GPU-accelerated max reduction using OpenCL
// - Prefers OpenCL 2.0 work-group reduction on Intel GPUs
// - Falls back to portable local-memory tree reduction on 1.2

#include <CL/cl.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

static void check(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "%s failed with error %d\n", msg, err);
        std::exit(1);
    }
}

static std::string load_text_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::string content;
    ifs.seekg(0, std::ios::end);
    content.resize(static_cast<size_t>(ifs.tellg()));
    ifs.seekg(0, std::ios::beg);
    ifs.read(&content[0], content.size());
    return content;
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    return (bool)f;
}

static std::string get_exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH] = {0};
    DWORD len = GetModuleFileNameA(nullptr, buf, (DWORD)sizeof(buf));
    if (len == 0 || len >= sizeof(buf)) return std::string();
    std::string full(buf, buf + len);
    size_t pos = full.find_last_of("/\\");
    if (pos == std::string::npos) return std::string();
    return full.substr(0, pos);
#else
    return std::string();
#endif
}

static std::string resolve_kernel_path() {
    // Try current working directory first
    const char* fname = "kernels.cl";
    std::vector<std::string> candidates;
    candidates.emplace_back(fname);

    std::string exedir = get_exe_dir();
    if (!exedir.empty()) {
        candidates.emplace_back(exedir + std::string("/") + fname);
    }
    // Fallback to typical source layout when running from repo root
    candidates.emplace_back("src/kernels.cl");

    for (const auto& p : candidates) {
        if (file_exists(p)) return p;
    }
    throw std::runtime_error("Failed to open file: " + std::string(fname));
}

static bool is_intel(const char* s) {
    if (!s) return false;
    std::string str(s);
    for (auto& c : str) c = (char)std::tolower((unsigned char)c);
    return str.find("intel") != std::string::npos;
}

static bool is_opencl_c_ge_20(cl_device_id dev) {
    // Prefer OpenCL C version query if available; fallback to device version.
    char buf[256] = {0};
    cl_int err = clGetDeviceInfo(dev, CL_DEVICE_OPENCL_C_VERSION, sizeof(buf), buf, nullptr);
    if (err != CL_SUCCESS) {
        // Fallback
        err = clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(buf), buf, nullptr);
        if (err != CL_SUCCESS) return false;
    }
    // Format: "OpenCL C 2.0 ..." or "OpenCL 3.0 ..."
    int major = 0, minor = 0;
    if (std::sscanf(buf, "OpenCL C %d.%d", &major, &minor) == 2) {
        return (major > 2) || (major == 2 && minor >= 0);
    }
    if (std::sscanf(buf, "OpenCL %d.%d", &major, &minor) == 2) {
        return (major > 2) || (major == 2 && minor >= 0);
    }
    return false;
}

struct Options {
    size_t size = 1 << 26; // default dataset size
    int wg = 256;          // work-group size
    int groups_max = 1024; // cap number of groups per pass
    unsigned seed = 42;    // RNG seed
    bool verbose = true;
    bool csv = false;      // emit CSV summary: size,kernel_ms,passes,wg,items
};

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto require_value = [&](int& i) {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + a);
        };
        if (a == "--size" || a == "-n") { require_value(i); opt.size = std::strtoull(argv[++i], nullptr, 10); }
        else if (a == "--wg") { require_value(i); opt.wg = std::atoi(argv[++i]); }
        else if (a == "--groups-max") { require_value(i); opt.groups_max = std::atoi(argv[++i]); }
        else if (a == "--seed") { require_value(i); opt.seed = (unsigned)std::strtoul(argv[++i], nullptr, 10); }
        else if (a == "--quiet" || a == "-q") { opt.verbose = false; }
        else if (a == "--csv") { opt.csv = true; }
        else if (a == "--help" || a == "-h") {
            std::cout << "Usage: ocl_find_max [--size N] [--wg W] [--groups-max G] [--seed S] [--quiet] [--csv]\n";
            std::exit(0);
        }
    }
    if (opt.wg <= 0) opt.wg = 256;
    if (opt.groups_max <= 0) opt.groups_max = 1024;
    return opt;
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_args(argc, argv);

        // Create data
        std::vector<float> data(opt.size);
        std::srand(opt.seed);
        for (size_t i = 0; i < data.size(); ++i) {
            // Spread across a range; include occasional NaN-safe values
            data[i] = ((float)std::rand() / RAND_MAX) * 1000.0f - 500.0f;
        }
        // Plant a clear maximum
        if (!data.empty()) data[data.size() / 2] = 123456.0f;

        // Pick Intel GPU if present
        cl_uint num_platforms = 0;
        check(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(query)");
        std::vector<cl_platform_id> plats(num_platforms);
        check(clGetPlatformIDs(num_platforms, plats.data(), nullptr), "clGetPlatformIDs(list)");

        cl_platform_id chosen_platform = nullptr;
        cl_device_id chosen_device = nullptr;
        for (auto p : plats) {
            cl_uint num_devices = 0;
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS || num_devices == 0) continue;
            std::vector<cl_device_id> devs(num_devices);
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_devices, devs.data(), nullptr) != CL_SUCCESS) continue;
            for (auto d : devs) {
                char vendor[256] = {0};
                clGetDeviceInfo(d, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
                if (is_intel(vendor)) { chosen_platform = p; chosen_device = d; break; }
            }
            if (chosen_device) break;
        }
        if (!chosen_device) {
            // Fallback: first GPU device anywhere
            for (auto p : plats) {
                cl_uint num_devices = 0;
                if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS || num_devices == 0) continue;
                std::vector<cl_device_id> devs(num_devices);
                if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, num_devices, devs.data(), nullptr) != CL_SUCCESS) continue;
                chosen_platform = p; chosen_device = devs[0];
                break;
            }
        }
        if (!chosen_device) {
            std::fprintf(stderr, "No OpenCL GPU device found.\n");
            return 1;
        }

        char name[256] = {0};
        clGetDeviceInfo(chosen_device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        char vendor[256] = {0};
        clGetDeviceInfo(chosen_device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
        if (opt.verbose) std::printf("Using device: %s (%s)\n", name, vendor);

        cl_int err = CL_SUCCESS;
        cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)chosen_platform, 0 };
        cl_context ctx = clCreateContext(props, 1, &chosen_device, nullptr, nullptr, &err);
        check(err, "clCreateContext");
        cl_command_queue q = clCreateCommandQueue(ctx, chosen_device, CL_QUEUE_PROFILING_ENABLE, &err);
        check(err, "clCreateCommandQueue");

        // Load kernel source (try cwd, exe dir, then src/)
        std::string kernel_path = resolve_kernel_path();
        std::string src = load_text_file(kernel_path);
        const char* src_ptr = src.c_str();
        size_t src_len = src.size();
        cl_program prog = clCreateProgramWithSource(ctx, 1, &src_ptr, &src_len, &err);
        check(err, "clCreateProgramWithSource");

        bool can_use_wg_reduce = is_opencl_c_ge_20(chosen_device);
        std::string build_opts;
        if (can_use_wg_reduce) {
            build_opts = "-cl-std=CL2.0 -DUSE_WG_REDUCE=1";
        } else {
            build_opts = "-cl-std=CL1.2";
        }
        err = clBuildProgram(prog, 1, &chosen_device, build_opts.c_str(), nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_sz = 0;
            clGetProgramBuildInfo(prog, chosen_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
            std::string log(log_sz, '\0');
            clGetProgramBuildInfo(prog, chosen_device, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
            std::fprintf(stderr, "Build failed. Options: %s\n%s\n", build_opts.c_str(), log.c_str());
            return 1;
        }

        cl_kernel krn = clCreateKernel(prog, "reduce_max_stage", &err);
        check(err, "clCreateKernel(reduce_max_stage)");

        // Buffers
        const size_t n = data.size();
        cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, data.data(), &err);
        check(err, "clCreateBuffer(A)");
        cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float) * n, nullptr, &err);
        check(err, "clCreateBuffer(B)");

        // Reduction loop
        const int wg = opt.wg; // 128 or 256 are good starting points on Intel iGPU
        const int ITEMS_PER_THREAD = 8; // tuning knob; 8–16 works well typically
        size_t in_count = n;
        bool use_A_as_input = true;
        uint64_t total_kernel_ns = 0;
        int pass_count = 0;

        auto launch_pass = [&](size_t count, cl_mem in_buf, cl_mem out_buf) -> size_t {
            // determine number of groups for this pass
            size_t groups = (count + (size_t)wg * ITEMS_PER_THREAD - 1) / ((size_t)wg * ITEMS_PER_THREAD);
            if (groups == 0) groups = 1;
            if ((int)groups > opt.groups_max) groups = (size_t)opt.groups_max;
            const size_t global = groups * (size_t)wg;

            cl_int e = 0;
            if (can_use_wg_reduce) {
                e = clSetKernelArg(krn, 0, sizeof(cl_mem), &in_buf);
                e |= clSetKernelArg(krn, 1, sizeof(cl_mem), &out_buf);
                cl_uint n_arg = (cl_uint)count;
                e |= clSetKernelArg(krn, 2, sizeof(cl_uint), &n_arg);
                check(e, "clSetKernelArg(wg)");
            } else {
                e = clSetKernelArg(krn, 0, sizeof(cl_mem), &in_buf);
                e |= clSetKernelArg(krn, 1, sizeof(cl_mem), &out_buf);
                cl_uint n_arg = (cl_uint)count;
                e |= clSetKernelArg(krn, 2, sizeof(cl_uint), &n_arg);
                // local memory scratch: one float per work-item
                e |= clSetKernelArg(krn, 3, sizeof(float) * (size_t)wg, nullptr);
                check(e, "clSetKernelArg(local)");
            }

            const size_t lsize = (size_t)wg;
            cl_event evt = nullptr;
            e = clEnqueueNDRangeKernel(q, krn, 1, nullptr, &global, &lsize, 0, nullptr, &evt);
            check(e, "clEnqueueNDRangeKernel");
            check(clWaitForEvents(1, &evt), "clWaitForEvents");
            cl_ulong t0 = 0, t1 = 0;
            check(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr), "clGetEventProfilingInfo(start)");
            check(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr), "clGetEventProfilingInfo(end)");
            if (t1 > t0) total_kernel_ns += (uint64_t)(t1 - t0);
            ++pass_count;
            clReleaseEvent(evt);
            return groups;
        };

        while (in_count > 1) {
            size_t out_count = launch_pass(in_count, use_A_as_input ? bufA : bufB, use_A_as_input ? bufB : bufA);
            in_count = out_count;
            use_A_as_input = !use_A_as_input;
        }

        // Read result back from the last output buffer
        float gpu_max = -std::numeric_limits<float>::infinity();
        cl_mem result_buf = use_A_as_input ? bufA : bufB; // last output buffer
        check(clEnqueueReadBuffer(q, result_buf, CL_TRUE, 0, sizeof(float), &gpu_max, 0, nullptr, nullptr), "clEnqueueReadBuffer(result)");

        // CPU verification
        float cpu_max = -std::numeric_limits<float>::infinity();
        for (float v : data) cpu_max = std::max(cpu_max, v);

        if (opt.verbose) {
            std::printf("GPU max: %.6f\n", gpu_max);
            std::printf("CPU max: %.6f\n", cpu_max);
        }
        const float diff = std::abs(gpu_max - cpu_max);
        if (diff > 1e-4f) {
            std::fprintf(stderr, "Mismatch detected: |GPU-CPU| = %g\n", diff);
            return 2;
        } else if (opt.verbose) {
            std::printf("Match.\n");
        }

        // Report GPU kernel timing (sum of all passes)
        const double kernel_ms = (double)total_kernel_ns / 1.0e6;
        if (opt.csv) {
            // CSV: size,kernel_ms,passes,wg,items_per_thread
            std::printf("%zu,%.6f,%d,%d,%d\n", n, kernel_ms, pass_count, wg, ITEMS_PER_THREAD);
        } else if (opt.verbose) {
            std::printf("Kernel passes: %d\n", pass_count);
            std::printf("Total kernel time: %.6f ms\n", kernel_ms);
        }

        // Cleanup
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseKernel(krn);
        clReleaseProgram(prog);
        clReleaseCommandQueue(q);
        clReleaseContext(ctx);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
