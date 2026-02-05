#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <random>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <numeric>
#define HALF
#ifdef HALF
#include "half.hpp"
#endif

#define LOCAL_SCRATCH 0
#define RAND_INT 0

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define HIP_CALL(call) do{  \
    hipError_t err = call;  \
    if(err != hipSuccess){  \
        printf("[hiperror](%d) fail to call %s",(int)err,#call);    \
        exit(0);            \
    }                       \
} while(0)

#define ABS(x) ((x) > 0 ? (x) : -(x))

using fp32_t = float;
using fp16_t = _Float16;
using float16 = half_float::half; // cpu type

using fp16x2_t = fp16_t __attribute__((ext_vector_type(2)));
using fp16x4_t = fp16_t __attribute__((ext_vector_type(4)));
using fp16x8_t = fp16_t __attribute__((ext_vector_type(8)));
using fp16x16_t = fp16_t __attribute__((ext_vector_type(16)));
using fp32x4_t = fp32_t __attribute__((ext_vector_type(4)));
using fp32x16_t = fp32_t __attribute__((ext_vector_type(16)));

using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
#define BUFFER_LOAD_DWORD3 0x00020000   // This is valid for 
struct buffer_resource {
    const void * ptr;
    uint32_t range;
    uint32_t config;
};
__device__ int32x4_t make_buffer_resource(const void * ptr, uint32_t size = 0xffffffff)
{
    buffer_resource res {ptr, size, BUFFER_LOAD_DWORD3};
    return __builtin_bit_cast(int32x4_t, res);
}
// A: M*K, B: N*K, C:M*N, use 16x16x16 fp16
/*
* V0/V1/   is 32bit register holding A/B matrix data, each register contains 2 fp16 pixel along gemm-k
* a0/a1... is 32bit register holding C matrix data in fp32 (this instruction use fp32 as acc)
* L0, L1.. is lane id with in a single wave, here we only have lane 0~63 (wave64)
* each thread need 2 registers for A, 2 regs for B, 16 regs for C

                                                            L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 10 11 12 13 14 15 
                                                 Matrix B   __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __
                                                           |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k0  L0~315
                                                           |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k1
                                                           |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k2
                                                          _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k3
                                                           |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k4  L15~31
                                                           |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k5
                                                           |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k6
                                                          _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k7
                                                           |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k4  L32~47
                                                           |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k5
                                                           |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k6
                                                          _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k7
                                                           |v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0|v0| k4  L48~63
                                                           |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__| k5
                                                           |v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1|v1| k6
                                                          _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|_k7                                                                                                                       

     Matrix A                                                                                  
     L0~15        L16~31      L32~47     L48~63             Matrix C      
     k0 k1|k2 k3|k4 k5|k6 k7|k8 k9|10 11|12 13|14 15          L0~L63      L0~L63      L0~L63      L0~L63
     _____|_____|_____|_____|_____|_____|_____|_____       |__ __ __ __|__ __ __ __|__ __ __ __|__ __ __ __ 
L0  |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| L0
L1  |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| L1
L2  |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| L2
L3  |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |     _|V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3|_L3
L4  |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| L4
L5  |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| L5
L6  |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| L6
L7  |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |     _|V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3|_L7
L8  |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| L8
L9  |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| L9
L10 |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| 10
L11 |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |     _|V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3|_11
L12 |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| 12
L13 |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| 13
L14 |v0   |v1   |v0   |v1   |v0   |v1   |v0   |v1   |      |V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3| 14
L15 |v0___|v1___|v0___|v1___|v0___|v1___|v0___|v1___|     _|V0|V0|V0|V0|V1|V1|V1|V1|V2|V2|V2|V2|V3|V3|V3|V3|_15
                                                                       |           |           |


*/
__global__ void 
matrix_core_kernel_standard(const void* __restrict__ ptr_a,
                   const void* __restrict__ ptr_b,
                   void* __restrict__ ptr_c,
                   int stride_a, // stride in unit of pixel
                   int stride_b,
                   int stride_c)
{
    // 16x16x16 gemm, assume only launced 1 wave
    int offset_a = (threadIdx.x / 16 * 4) + (threadIdx.x % 16 * stride_a);
    int offset_b = (threadIdx.x / 16 * 4) + (threadIdx.x % 16 * stride_b);

    fp16x4_t v_a = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_a) + offset_a);
    fp16x4_t v_b = *reinterpret_cast<const fp16x4_t*>(reinterpret_cast<const fp16_t*>(ptr_b) + offset_b);
    fp32x4_t v_c = {.0f};  // clear

    v_c = __builtin_hcu_mmac_f32_16x16x16_f16(v_a, v_b, v_c);

    fp16x16_t v_c_f16;
    for(auto i = 0; i < 16; i++) {
        v_c_f16[i] = static_cast<fp16_t>(v_c[i]);
    }

    int row_id_c = threadIdx.x % 16;
    int col_id_c = threadIdx.x / 16;
    int offset_c = row_id_c * stride_c + col_id_c;

    for(auto i = 0; i < 4; i++) {
        // int row_offset = (i % 4) + (i / 4 * 8);
        int col_shift = i * 4;
        *(reinterpret_cast<fp16_t*>(ptr_c) + offset_c + col_shift) = v_c_f16[i];
    }
}

#ifdef RAND_INT
#define PER_PIXEL_CHECK
#endif

static inline bool valid_vector( const float* ref, const float16* pred, int n, double nrms = 1e-3 )
{    
    double s0=0.0;
    double s1=0.0;
#ifdef PER_PIXEL_CHECK
    int pp_err = 0;
#endif
    int i_start = 0, i_end=n;
    
    for( int i=i_start; i<i_end; ++i ){
        double ri=(double)ref[i];
        double pi=(double)pred[i];
        double d=ri-pi;
        double dd=d*d;
        double rr=2.0*ri*ri;
        s0+=dd;
        s1+=rr;
        
#ifdef PER_PIXEL_CHECK
        double delta = ABS(ri-pi)/ri;
        if(delta>1e-3){
            if(pp_err<100)
                printf("diff at %4d, ref:%lf, pred:%lf(0x%04x), d:%lf\n",i,ri,pi,((uint16_t*)pred)[i],delta);
            pp_err++;
        }
#endif
    }
    // int i_num = i_end - i_start;
    // printf("pp_crr:%d, pp_err:%d, crr_ratio:%.3f, nrms:%lf, s0:%lf, s1:%lf\n",i_num-pp_err, pp_err, (float)(i_num-pp_err)/(float)i_num, sqrt(s0/s1),s0,s1);

    return (sqrt(s0/s1)<nrms)
#ifdef PER_PIXEL_CHECK
        && (pp_err==0)
#endif
    ;
}

void rand_vector_2d(float* v, int row, int col, int ld, float min_v = 0, float max_v = 1){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            float tmp = float(std::rand()) / float(RAND_MAX);
            v[r*ld+c] = static_cast<float>(min_v + tmp * (max_v - min_v));
            // v[r*ld+c] =   ((float)(r*ld+c)) / (row/2 * col/2) - 5;
        }
    }
}

void rand_vector_2d_int(float* v, int row, int col, int ld){
    int r,c;
    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }
    for(r=0;r<row;r++){
        for(c=0;c<col;c++){
            v[r*ld+c] = ((float)(rand() % 10)) - 5;
        }
    }
}

void gemm_rcr(
    const float*  __restrict__ ptr_a,
    const float*  __restrict__ ptr_b,
    float*  ptr_c,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc)
{
    for(auto i_m = 0 ; i_m < m; i_m++) {
        for(auto i_n = 0; i_n < n; i_n++) {
            float acc = 0;
            for(auto i_k = 0; i_k < k; i_k++) {
                acc += ptr_a[i_m * lda + i_k] * ptr_b[i_n * ldb + i_k];
            }
            ptr_c[i_m * ldc + i_n] = acc;
        }
    }
}

int main(int argc, char ** argv)
{
    int m = 16;
    int n = 16;
    int k = 16;

    int lda = k;
    int ldb = k;
    int ldc = n;

    float *host_a, *host_b, *host_c;
    float16 *fp16_a, *fp16_b, *fp16_c, *dev_a, *dev_b, *dev_c;

    //fp32 on host
    host_a = (float*)malloc(lda*m*sizeof(float));
    host_b = (float*)malloc(ldb*n*sizeof(float));
    host_c = (float*)malloc(ldc*m*sizeof(float));

#ifdef RAND_INT
    rand_vector_2d_int(host_a, m, k, lda);
    rand_vector_2d_int(host_b, n, k, ldb);
#else
    rand_vector_2d(host_a, m, k, lda, 0.0, 1.0);
    rand_vector_2d(host_b, n, k, ldb, -0.5, 0.5);
#endif

    //fp16 on host
    fp16_a = (float16*)malloc(lda*m*sizeof(float16));
    fp16_b = (float16*)malloc(ldb*n*sizeof(float16));
    fp16_c = (float16*)malloc(ldc*m*sizeof(float16));
    
    //convert fp32 a and b into fp16 on host
    for(int i=0; i<lda*m; i++)fp16_a[i]=__float2half_rn(host_a[i]);
    for(int i=0; i<ldb*n; i++)fp16_b[i]=__float2half_rn(host_b[i]);

    HIP_CALL(hipMalloc(&dev_a, lda*m*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_b, ldb*n*sizeof(float16)));
    HIP_CALL(hipMalloc(&dev_c, ldc*m*sizeof(float16)));
    //fp16 cpy to device
    HIP_CALL(hipMemcpy(dev_a, fp16_a, lda*m*sizeof(float16), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dev_b, fp16_b, ldb*n*sizeof(float16), hipMemcpyHostToDevice));

    printf("m:%d,n:%d,k:%d,lda:%d,ldb:%d,ldc:%d\n",  m, n, k, lda, ldb, ldc); fflush(stdout);
    gemm_rcr(host_a, host_b, host_c, m,n,k,lda,ldb,ldc);

#if 0
    printf("print matrix A:\n");
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<k; j++)
        {
            printf("%.2f ", static_cast<float>(fp16_a[i*k+j]));
        }
        printf("\n");
    }
    printf("\n");

    printf("printf matrix B:\n");
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<k; j++)
        {
            printf("%.2f ", static_cast<float>(fp16_b[i*k+j]));
        }
        printf("\n");
    }
    printf("\n");

    printf("print matrix C:\n");
    for(int i=0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            printf("%.2f ", host_c[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");

#endif

    {
        matrix_core_kernel_standard<<<1, 64>>>(dev_a, dev_b, dev_c, lda, ldb, ldc);

        HIP_CALL(hipMemcpy(fp16_c, dev_c, ldc*m*sizeof(float16), hipMemcpyDeviceToHost));
        bool res = valid_vector( host_c, fp16_c, m*n, 1e-3);
        printf("[16x16x16, standard], %s",res?"valid":"fail");fflush(stdout);
        printf("\n"); fflush(stdout);
    }

#if 0
    printf("print device C:\n");
    for(int i=0; i<m; i++)
    {
        for(int j=0;j<n; j++)
        {
            printf("%.2f ", static_cast<float>(fp16_c[i*n+j]));
        }
        printf("\n");
    }
    printf("\n");
#endif
    
    free(host_a);
    free(host_b);
    free(host_c);
    free(fp16_a);
    free(fp16_b);
    free(fp16_c);
    
    HIP_CALL(hipFree(dev_a));
    HIP_CALL(hipFree(dev_b));
    HIP_CALL(hipFree(dev_c));
}
