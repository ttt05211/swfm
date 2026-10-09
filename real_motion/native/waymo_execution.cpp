// Separate execution ABI: never change the existing column kernels/cache.
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
extern "C" { int _fltused = 0; }
using u8 = unsigned char;
extern "C" void* memset(void*, int, decltype(sizeof(0)));
#pragma function(memset)
extern "C" void* memset(void* dst, int value, decltype(sizeof(0)) n) {
    volatile u8* p = static_cast<volatile u8*>(dst);
    for (decltype(n) i=0; i<n; ++i) p[i]=static_cast<u8>(value);
    return dst;
}
#else
#define API extern "C" __attribute__((visibility("default")))
using u8 = unsigned char;
#endif
using i64 = long long;
API int swfm_waymo_execution_abi() noexcept { return 1; }

// np.lexsort((distance,flat)) is stable: earliest input wins an equal tie.
// Coordinates/distances are computed by the ORIGINAL full-batch FP64 NumPy.
API int swfm_warp_winners(const i64* flat, const double* distance,
    const u8* labels, i64 n, i64 volume, int free_label, u8* out, double* best) noexcept {
    for (i64 j=0; j<volume; ++j) { out[j]=static_cast<u8>(free_label); best[j]=1.7976931348623157e308; }
    for (i64 j=0; j<n; ++j) {
        const i64 f=flat[j];
        if (f<0 || f>=volume || !(distance[j]>=0.)) return 1;
        if (distance[j]<best[f]) { best[f]=distance[j]; out[f]=labels[j]; }
    }
    return 0;
}

// NumPy's contiguous length-16 float64 pairwise sum (eight accumulators).
static double sum16(const double* a) noexcept {
    double r[8];
    for (int j=0; j<8; ++j) r[j]=a[j]+a[j+8];
    return ((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]));
}

// No fast-math/FMA, no changed fit, no floating coordinate transform.
// Strided [16,3] mean reduction is sequential; scalar [16] covariances pairwise.
// sqrt/clip/final FP32 conversion remain NumPy operations in the caller.
API int swfm_surface_fit(const double* delta, const double* weight,
    const double* distance, const double* recent, const double* opposite,
    const u8* opposite_seen, i64 n, double* out) noexcept {
    for (i64 row=0; row<n; ++row) {
        const double* d=delta+row*48; const double* w=weight+row*16;
        double total=sum16(w); if (total<1e-12) total=1e-12;
        double mean[3]={0.,0.,0.}; int count=0, column=0;
        for (int j=0; j<16; ++j) {
            for (int k=0; k<3; ++k) mean[k]+=w[j]*d[j*3+k];
            if (w[j]>0.) {
                ++count;
                if (d[j*3]>-.55 && d[j*3]<.55 && d[j*3+1]>-.55 && d[j*3+1]<.55) ++column;
            }
        }
        for (int k=0; k<3; ++k) mean[k]/=total;
        double c[48]; for (int j=0; j<16; ++j) for (int k=0; k<3; ++k) c[j*3+k]=d[j*3+k]-mean[k];
        double cov[6], scratch[16]; const int ax[6]={0,1,0,0,1,2}, bx[6]={0,1,1,2,2,2};
        for (int k=0; k<6; ++k) {
            for (int j=0; j<16; ++j) scratch[j]=(w[j]*c[j*3+ax[k]])*c[j*3+bx[k]];
            cov[k]=sum16(scratch)/total;
        }
        const double xx=cov[0]+1e-3, yy=cov[1]+1e-3, xy=cov[2], xz=cov[3], yz=cov[4];
        double determinant=xx*yy-xy*xy; if (determinant<1e-9) determinant=1e-9;
        const double gx=(xz*yy-yz*xy)/determinant, gy=(yz*xx-xz*xy)/determinant;
        const double intercept=(mean[2]-gx*mean[0])-gy*mean[1];
        for (int j=0; j<16; ++j) {
            const double error=d[j*3+2]-((intercept+gx*d[j*3])+gy*d[j*3+1]);
            scratch[j]=w[j]*(error*error);
        }
        double* r=out+row*12;
        r[0]=count>=3; r[1]=-intercept; r[2]=sum16(scratch)/total;
        r[3]=cov[5]>0. ? cov[5] : 0.; r[4]=gx; r[5]=gy; r[6]=count/16.;
        r[7]=count>0 ? distance[row*16]/2.5 : 1.; r[8]=opposite[row]; r[9]=opposite_seen[row];
        r[10]=static_cast<double>(column)/(count>0 ? count : 1);
        for (int j=0; j<16; ++j) scratch[j]=w[j]*recent[row*16+j];
        r[11]=sum16(scratch)/total;
    }
    return 0;
}
