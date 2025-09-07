// Max reduction kernel with two variants:
// - Fast path (OpenCL 2.0+): uses work_group_reduce_max
// - Portable path (OpenCL 1.2): tree reduction in local memory
// Host compiles with -DUSE_WG_REDUCE=1 when OpenCL C >= 2.0

#ifdef USE_WG_REDUCE
// Requires OpenCL C 2.0 or newer
__kernel void reduce_max_stage(
    __global const float* in,
    __global float* out,
    const uint n)
{
    const size_t gid = get_global_id(0);
    const size_t gsize = get_global_size(0);

    float acc = -INFINITY;
    for (size_t i = gid; i < (size_t)n; i += gsize) {
        float v = in[i];
        acc = fmax(acc, v);
    }

    // Work-group reduction to a single max
    float wg_max = work_group_reduce_max(acc);
    if (get_local_id(0) == 0) {
        out[get_group_id(0)] = wg_max;
    }
}

#else
// Portable OpenCL 1.2-compatible kernel
__kernel void reduce_max_stage(
    __global const float* in,
    __global float* out,
    const uint n,
    __local float* scratch)
{
    const size_t lid = get_local_id(0);
    const size_t gid = get_global_id(0);
    const size_t gsize = get_global_size(0);

    float acc = -INFINITY;
    for (size_t i = gid; i < (size_t)n; i += gsize) {
        float v = in[i];
        acc = fmax(acc, v);
    }

    scratch[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction in local memory
    for (uint stride = get_local_size(0) >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = fmax(scratch[lid], scratch[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        out[get_group_id(0)] = scratch[0];
    }
}
#endif

