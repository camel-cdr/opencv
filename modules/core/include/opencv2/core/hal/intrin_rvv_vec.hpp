
#ifndef OPENCV_HAL_INTRIN_RVV_VEC_HPP
#define OPENCV_HAL_INTRIN_RVV_VEC_HPP

// #include <riscv_vector.h>
#include <initializer_list>
#include <assert.h>
#include <vector>

namespace cv
{
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN
namespace rvv
{
#define CV_SIMD128 1
#define CV_SIMD128_64F 1
#define RVV_VLEN_MAX 65536

using v_uint8 = vuint8m1_t;
using v_int8 = vint8m1_t;
using v_uint16 = vuint16m1_t;
using v_int16 = vint16m1_t;
using v_uint32 = vuint32m1_t;
using v_int32 = vint32m1_t;
using v_uint64 = vuint64m1_t;
using v_int64 = vint64m1_t;

using v_float32 = vfloat32m1_t;
#if CV_SIMD128_64F
using v_float64 = vfloat64m1_t;
#endif

using uchar = unsigned char;
using schar = signed char;
using ushort = unsigned short;
using uint = unsigned int;
using uint64 = unsigned long int;
using int64 = long int;



template <class T>
struct VTraits;

template <>
struct VTraits<v_uint8> 
{
    static int nlanes; 
    using lane_type = uchar;
};

template <>
struct VTraits<v_int8> 
{
    static int nlanes; 
    using lane_type = schar;
};
template <>
struct VTraits<v_uint16> 
{
    static int nlanes; 
    using lane_type = ushort;
};
template <>
struct VTraits<v_int16> 
{
    static int nlanes; 
    using lane_type = short;
};
template <>
struct VTraits<v_uint32> 
{
    static int nlanes; 
    using lane_type = uint;
};
template <>
struct VTraits<v_int32> 
{
    static int nlanes; 
    using lane_type = int;
};

template <>
struct VTraits<v_float32> 
{
    static int nlanes; 
    using lane_type = float;
};
template <>
struct VTraits<v_uint64> 
{
    static int nlanes; 
    using lane_type = uint64;
};
template <>
struct VTraits<v_int64> 
{
    static int nlanes; 
    using lane_type = int64;
};
#if CV_SIMD128_64F
template <>
struct VTraits<v_float64> 
{
    static int nlanes; 
    using lane_type = double;
};
#endif

inline int VTraits<v_uint8>::nlanes = vsetvlmax_e8m1();
inline int VTraits<v_int8>::nlanes = vsetvlmax_e8m1();
inline int VTraits<v_uint16>::nlanes = vsetvlmax_e16m1();
inline int VTraits<v_int16>::nlanes = vsetvlmax_e16m1();
inline int VTraits<v_float32>::nlanes = vsetvlmax_e32m1();
inline int VTraits<v_uint32>::nlanes = vsetvlmax_e32m1();
inline int VTraits<v_int32>::nlanes = vsetvlmax_e32m1();
inline int VTraits<v_uint64>::nlanes = vsetvlmax_e64m1();
inline int VTraits<v_int64>::nlanes = vsetvlmax_e64m1();
#if CV_SIMD128_64F
inline int VTraits<v_float64>::nlanes = vsetvlmax_e64m1();
#endif


//////////// get0 ////////////
#define OPENCV_HAL_IMPL_RVV_GRT0_INT(_Tpvec, _Tp) \
inline _Tp v_get0(v_##_Tpvec v) \
{ \
    return vmv_x(v); \
}

OPENCV_HAL_IMPL_RVV_GRT0_INT(uint8, uchar)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int8, schar)
OPENCV_HAL_IMPL_RVV_GRT0_INT(uint16, ushort)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int16, short)
OPENCV_HAL_IMPL_RVV_GRT0_INT(uint32, unsigned)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int32, int)
OPENCV_HAL_IMPL_RVV_GRT0_INT(uint64, uint64)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int64, int64)

inline float v_get0(v_float32 v) \
{ \
    return vfmv_f(v); \
}
#if CV_SIMD128_64F
inline double v_get0(v_float64 v) \
{ \
    return vfmv_f(v); \
}
#endif

//////////// Initial ////////////

#define OPENCV_HAL_IMPL_RVV_INIT_INTEGER(_Tpvec, _Tp, suffix1, suffix2, vl) \
inline v_##_Tpvec v_setzero_##suffix1() \
{ \
    return vmv_v_x_##suffix2##m1(0, vl); \
} \
inline v_##_Tpvec v_setall_##suffix1(_Tp v) \
{ \
    return vmv_v_x_##suffix2##m1(v, vl); \
}

OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint8, uchar, u8, u8, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int8, schar, s8, i8, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint16, ushort, u16, u16, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int16, short, s16, i16, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint32, uint, u32, u32, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int32, int, s32, i32, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint64, uint64, u64, u64, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int64, int64, s64, i64, VTraits<v_int64>::nlanes)

#define OPENCV_HAL_IMPL_RVV_INIT_FP(_Tpv, _Tp, suffix, vl) \
inline v_##_Tpv v_setzero_##suffix() \
{ \
    return vfmv_v_f_##suffix##m1(0, vl); \
} \
inline v_##_Tpv v_setall_##suffix(_Tp v) \
{ \
    return vfmv_v_f_##suffix##m1(v, vl); \
}

OPENCV_HAL_IMPL_RVV_INIT_FP(float32, float, f32, VTraits<v_float32>::nlanes)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_INIT_FP(float64, double, f64, VTraits<v_float64>::nlanes)
#endif

//////////// Reinterpret ////////////
// TODO: can be simplified by using overloaded RV intrinsic
#define OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(_Tpvec1, _Tpvec2, suffix1, suffix2, nsuffix1, nsuffix2) \
inline v_##_Tpvec1 v_reinterpret_as_##suffix1(const v_##_Tpvec2& v) \
{ \
    return v_##_Tpvec1(vreinterpret_v_##nsuffix2##m1_##nsuffix1##m1(v));\
} \
inline v_##_Tpvec2 v_reinterpret_as_##suffix2(const v_##_Tpvec1& v) \
{ \
    return v_##_Tpvec2(vreinterpret_v_##nsuffix1##m1_##nsuffix2##m1(v));\
}

OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, int8, u8, s8, u8, i8)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint16, int16, u16, s16, u16, i16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint32, int32, u32, s32, u32, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint32, float32, u32, f32, u32, f32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int32, float32, s32, f32, i32, f32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint64, int64, u64, s64, u64, i64)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint64, float64, u64, f64, u64, f64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int64, float64, s64, f64, i64, f64)
#endif
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, uint16, u8, u16, u8, u16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, uint32, u8, u32, u8, u32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, uint64, u8, u64, u8, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint16, uint32, u16, u32, u16, u32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint16, uint64, u16, u64, u16, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint32, uint64, u32, u64, u32, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int8, int16, s8, s16, i8, i16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int8, int32, s8, s32, i8, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int8, int64, s8, s64, i8, i64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int16, int32, s16, s32, i16, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int16, int64, s16, s64, i16, i64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int32, int64, s32, s64, i32, i64)


#define OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(_Tpvec1, _Tpvec2, suffix1, suffix2, nsuffix1, nsuffix2, width1, width2) \
inline v_##_Tpvec1 v_reinterpret_as_##suffix1(const v_##_Tpvec2& v) \
{ \
    return vreinterpret_v_##nsuffix1##width2##m1_##nsuffix1##width1##m1(vreinterpret_v_##nsuffix2##width2##m1_##nsuffix1##width2##m1(v));\
} \
inline v_##_Tpvec2 v_reinterpret_as_##suffix2(const v_##_Tpvec1& v) \
{ \
    return vreinterpret_v_##nsuffix1##width2##m1_##nsuffix2##width2##m1(vreinterpret_v_##nsuffix1##width1##m1_##nsuffix1##width2##m1(v));\
}

OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, int16, u8, s16, u, i, 8, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, int32, u8, s32, u, i, 8, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, int64, u8, s64, u, i, 8, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, int8, u16, s8, u, i, 16, 8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, int32, u16, s32, u, i, 16, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, int64, u16, s64, u, i, 16, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, int8, u32, s8, u, i, 32, 8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, int16, u32, s16, u, i, 32, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, int64, u32, s64, u, i, 32, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, int8, u64, s8, u, i, 64, 8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, int16, u64, s16, u, i, 64, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, int32, u64, s32, u, i, 64, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, float32, u8, f32, u, f, 8, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, float32, u16, f32, u, f, 16, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, float32, u64, f32, u, f, 64, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int8, float32, s8, f32, i, f, 8, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int16, float32, s16, f32, i, f, 16, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int64, float32, s64, f32, i, f, 64, 32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, float64, u8, f64, u, f, 8, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, float64, u16, f64, u, f, 16, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, float64, u32, f64, u, f, 32, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int8, float64, s8, f64, i, f, 8, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int16, float64, s16, f64, i, f, 16, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int32, float64, s32, f64, i, f, 32, 64)
// Three times reinterpret
inline v_float32 v_reinterpret_as_f32(const v_float64& v) \
{ \
    return vreinterpret_v_u32m1_f32m1(vreinterpret_v_u64m1_u32m1(vreinterpret_v_f64m1_u64m1(v)));\
}

inline v_float64 v_reinterpret_as_f64(const v_float32& v) \
{ \
    return vreinterpret_v_u64m1_f64m1(vreinterpret_v_u32m1_u64m1(vreinterpret_v_f32m1_u32m1(v)));\
}
#endif

//////////// Extract //////////////

#define OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(_Tpvec, _Tp, suffix, vl) \
template <int s = 0> \
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b, int i = s) \
{ \
    return vslideup(vslidedown(v_setzero_##suffix(), a, i, vl), b, VTraits<_Tpvec>::nlanes - i, vl); \
} \
template<int s = 0> inline _Tp v_extract_n(_Tpvec v, int i = s) \
{ \
    return vmv_x(vslidedown(v_setzero_##suffix(), v, i, vl)); \
}


OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint8, uchar, u8, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int8, schar, s8, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint16, ushort, u16, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int16, short, s16, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint32, unsigned int, u32, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int32, int, s32, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint64, uint64, u64, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int64, int64, s64, VTraits<v_int64>::nlanes)

#define OPENCV_HAL_IMPL_RVV_EXTRACT_FP(_Tpvec, _Tp, suffix, vl) \
template <int s = 0> \
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b, int i = s) \
{ \
    return vslideup(vslidedown(v_setzero_##suffix(), a, i, vl), b, VTraits<_Tpvec>::nlanes - i, vl); \
} \
template<int s = 0> inline _Tp v_extract_n(_Tpvec v, int i = s) \
{ \
    return vfmv_f(vslidedown(v_setzero_##suffix(), v, i, vl)); \
}

OPENCV_HAL_IMPL_RVV_EXTRACT_FP(v_float32, float, f32, VTraits<v_float32>::nlanes)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_EXTRACT_FP(v_float64, double, f64, VTraits<v_float64>::nlanes)
#endif

////////////// Load/Store //////////////
// #include <stdarg.h>
// inline v_int32 v_load(int a, ...)
// {
//     return v_load(a, va_list);
// }

// inline v_int16 v_load(std::initializer_list<short> nScalars) \
// { \
//     return vle16_v_i16m1(nScalars.begin(), nScalars.size()); \
// }

#define OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(_Tpvec, _nTpvec, _Tp, hvl, vl, width, suffix, vmv) \
inline _Tpvec v_load(const _Tp* ptr) \
{ \
    return vle##width##_v_##suffix##m1(ptr, vl); \
} \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ \
    return vle##width##_v_##suffix##m1(ptr, vl); \
} \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ \
    return vle##width##_v_##suffix##m1(ptr, hvl); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ \
    vse##width(ptr, a, vl); \
} \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ \
    vse##width(ptr, a, vl); \
} \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ \
    vse##width(ptr, a, vl); \
} \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ \
    vse##width(ptr, a, hvl); \
} \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    vse##width(ptr, vslidedown_vx_##suffix##m1(vmv(0, vl), a, hvl, vl), hvl); \
} \
inline _Tpvec v_load(std::initializer_list<_Tp> nScalars) \
{ \
    assert(nScalars.size() == vl); \
    return vle##width##_v_##suffix##m1(nScalars.begin(), nScalars.size()); \
} \
template<typename... Targs> \
_Tpvec v_load_##suffix(Targs... nScalars) \
{ \
    return v_load({nScalars...}); \
}
// inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode /*mode*/) \
// { \
//     vse##width##_v_##suffix##m1(ptr, a, vl); \
// } \

OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint8, vuint8m1_t, uchar, VTraits<v_uint8>::nlanes / 2, VTraits<v_uint8>::nlanes, 8, u8, vmv_v_x_u8m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int8, vint8m1_t, schar, VTraits<v_int8>::nlanes / 2, VTraits<v_int8>::nlanes, 8, i8, vmv_v_x_i8m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint16, vuint16m1_t, ushort, VTraits<v_uint16>::nlanes / 2, VTraits<v_uint16>::nlanes, 16, u16, vmv_v_x_u16m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int16, vint16m1_t, short, VTraits<v_int16>::nlanes / 2, VTraits<v_int16>::nlanes, 16, i16, vmv_v_x_i16m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint32, vuint32m1_t, unsigned int, VTraits<v_uint32>::nlanes / 2, VTraits<v_uint32>::nlanes, 32, u32, vmv_v_x_u32m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int32, vint32m1_t, int, VTraits<v_int32>::nlanes / 2, VTraits<v_int32>::nlanes, 32, i32, vmv_v_x_i32m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint64, vuint64m1_t, uint64, VTraits<v_uint64>::nlanes / 2, VTraits<v_uint64>::nlanes, 64, u64, vmv_v_x_u64m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int64, vint64m1_t, int64, VTraits<v_int64>::nlanes / 2, VTraits<v_int64>::nlanes, 64, i64, vmv_v_x_i64m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_float32, vfloat32m1_t, float, VTraits<v_float32>::nlanes /2 , VTraits<v_float32>::nlanes, 32, f32, vfmv_v_f_f32m1)

#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_float64, vfloat64m1_t, double, VTraits<v_float64>::nlanes / 2, VTraits<v_float64>::nlanes, 64, f64, vfmv_v_f_f64m1)
#endif

inline v_int8 v_load_halves(const schar* ptr0, const schar* ptr1)
{
    schar elems[16] =
    {
        ptr0[0], ptr0[1], ptr0[2], ptr0[3], ptr0[4], ptr0[5], ptr0[6], ptr0[7],
        ptr1[0], ptr1[1], ptr1[2], ptr1[3], ptr1[4], ptr1[5], ptr1[6], ptr1[7]
    };
    return vle8_v_i8m1(elems, 16);
}
inline v_uint8 v_load_halves(const uchar* ptr0, const uchar* ptr1) { return v_reinterpret_as_u8(v_load_halves((schar*)ptr0, (schar*)ptr1)); }

inline v_int16 v_load_halves(const short* ptr0, const short* ptr1)
{
    short elems[8] =
    {
        ptr0[0], ptr0[1], ptr0[2], ptr0[3], ptr1[0], ptr1[1], ptr1[2], ptr1[3]
    };
    return vle16_v_i16m1(elems, 8);
}
inline v_uint16 v_load_halves(const ushort* ptr0, const ushort* ptr1) { return v_reinterpret_as_u16(v_load_halves((short*)ptr0, (short*)ptr1)); }

inline v_int32 v_load_halves(const int* ptr0, const int* ptr1)
{
    int elems[4] =
    {
        ptr0[0], ptr0[1], ptr1[0], ptr1[1]
    };
    return vle32_v_i32m1(elems, 4);
}
inline v_float32 v_load_halves(const float* ptr0, const float* ptr1)
{
    float elems[4] =
    {
        ptr0[0], ptr0[1], ptr1[0], ptr1[1]
    };
    return vle32_v_f32m1(elems, 4);
}
inline v_uint32 v_load_halves(const unsigned* ptr0, const unsigned* ptr1) { return v_reinterpret_as_u32(v_load_halves((int*)ptr0, (int*)ptr1)); }

inline v_int64 v_load_halves(const int64* ptr0, const int64* ptr1)
{
    int64 elems[2] =
    {
        ptr0[0], ptr1[0]
    };
    return vle64_v_i64m1(elems, 2);
}
inline v_uint64 v_load_halves(const uint64* ptr0, const uint64* ptr1) { return v_reinterpret_as_u64(v_load_halves((int64*)ptr0, (int64*)ptr1)); }

#if CV_SIMD128_64F
inline v_float64 v_load_halves(const double* ptr0, const double* ptr1)
{
    double elems[2] =
    {
        ptr0[0], ptr1[0]
    };
    return vle64_v_f64m1(elems, 2);
}
#endif

////////////// Lookup table access ////////////////////
#define OPENCV_HAL_IMPL_RVV_LUT(_Tpvec, _Tp, suffix) \
inline _Tpvec v_lut(const _Tp* tab, const int* idx) \
{ \
    vuint32##suffix##_t vidx = vmul(vreinterpret_u32##suffix(vle32_v_i32##suffix(idx, VTraits<_Tpvec>::nlanes)), sizeof(_Tp), VTraits<_Tpvec>::nlanes); \
    return vloxei32(tab, vidx, VTraits<_Tpvec>::nlanes); \
} \
inline _Tpvec v_lut_pairs(const _Tp* tab, const int* idx) \
{ \
    std::vector<uint> idx_; \
    for (size_t i = 0; i < VTraits<v_int16>::nlanes; ++i) { \
        idx_.push_back(idx[i]); \
        idx_.push_back(idx[i]+1); \
    } \
    vuint32##suffix##_t vidx = vmul(vle32_v_u32##suffix(idx_.data(), VTraits<_Tpvec>::nlanes), sizeof(_Tp), VTraits<_Tpvec>::nlanes); \
    return vloxei32(tab, vidx, VTraits<_Tpvec>::nlanes); \
} \
inline _Tpvec v_lut_quads(const _Tp* tab, const int* idx) \
{ \
    std::vector<uint> idx_; \
    for (size_t i = 0; i < VTraits<v_int32>::nlanes; ++i) { \
        idx_.push_back(idx[i]); \
        idx_.push_back(idx[i]+1); \
        idx_.push_back(idx[i]+2); \
        idx_.push_back(idx[i]+3); \
    } \
    vuint32##suffix##_t vidx = vmul(vle32_v_u32##suffix(idx_.data(), VTraits<_Tpvec>::nlanes), sizeof(_Tp), VTraits<_Tpvec>::nlanes); \
    return vloxei32(tab, vidx, VTraits<_Tpvec>::nlanes); \
}
OPENCV_HAL_IMPL_RVV_LUT(v_int8, schar, m4)
OPENCV_HAL_IMPL_RVV_LUT(v_int16, short, m2)
OPENCV_HAL_IMPL_RVV_LUT(v_int32, int, m1)
OPENCV_HAL_IMPL_RVV_LUT(v_int64, int64_t, mf2)

inline v_uint8 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((schar*)tab, idx)); }
inline v_uint8 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((schar*)tab, idx)); }
inline v_uint8 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((schar*)tab, idx)); }
inline v_uint16 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((short*)tab, idx)); }
inline v_uint16 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((short*)tab, idx)); }
inline v_uint16 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((short*)tab, idx)); }
inline v_uint32 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((int*)tab, idx)); }
inline v_uint32 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((int*)tab, idx)); }
inline v_uint32 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((int*)tab, idx)); }
inline v_uint64 v_lut(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
inline v_uint64 v_lut_pairs(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }
inline v_uint64 v_lut_quads(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_quads((const int64_t*)tab, idx)); }

////////////// Pack boolean ////////////////////
inline v_uint8 v_pack_b(const v_uint16& a, const v_uint16& b)
{
    return vnsrl(vset(vlmul_ext_u16m2(a),1,b), 0, VTraits<v_uint8>::nlanes);
}

inline v_uint8 v_pack_b(const v_uint32& a, const v_uint32& b,
                           const v_uint32& c, const v_uint32& d)
{
    
    return vnsrl(vnsrl(vset(vset(vset(vlmul_ext_u32m4(a),1,b),2,c),3,d), 0, VTraits<v_uint8>::nlanes), 0, VTraits<v_uint8>::nlanes);
}

inline v_uint8 v_pack_b(const v_uint64& a, const v_uint64& b, const v_uint64& c,
                           const v_uint64& d, const v_uint64& e, const v_uint64& f,
                           const v_uint64& g, const v_uint64& h)
{
    return vnsrl(vnsrl(vnsrl(
        vset(vset(vset(vset(vset(vset(vset(vlmul_ext_u64m8(a),
        1,b),2,c),3,d),4,e),5,f),6,g),7,h), 
        0, VTraits<v_uint8>::nlanes), 0, VTraits<v_uint8>::nlanes), 0, VTraits<v_uint8>::nlanes);
}
////////////// Arithmetics (wrap)//////////////
#define OPENCV_HAL_IMPL_RVV_BIN_OP(_Tpvec, vl) \
inline _Tpvec v_add(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vadd(a, b, vl); \
} \
inline _Tpvec v_sub(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vsub(a, b, vl); \
} \
inline _Tpvec v_and(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vand(a, b, vl); \
} \
inline _Tpvec v_or(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vor(a, b, vl); \
} \
inline _Tpvec v_xor(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vxor(a, b, vl); \
}

OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint32, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int32, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint64, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int64, VTraits<v_int64>::nlanes)

#define OPENCV_HAL_IMPL_RVV_BIN_MUL(_Tpvec, vl) \
inline _Tpvec v_mul(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vmul(a, b, vl); \
}

OPENCV_HAL_IMPL_RVV_BIN_MUL(v_uint32, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_MUL(v_int32, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_MUL(v_uint64, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_MUL(v_int64, VTraits<v_int64>::nlanes)

#define OPENCV_HAL_IMPL_RVV_DIV_INT(_Tpvec, vl) \
inline _Tpvec v_div(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vdiv(a, b, vl); \
}
#define OPENCV_HAL_IMPL_RVV_DIV_UINT(_Tpvec, vl) \
inline _Tpvec v_div(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vdivu(a, b, vl); \
}
OPENCV_HAL_IMPL_RVV_DIV_UINT(v_uint8, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_DIV_INT(v_int8, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_DIV_UINT(v_uint16, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_DIV_INT(v_int16, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_DIV_UINT(v_uint32, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_DIV_INT(v_int32, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_DIV_UINT(v_uint64, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_DIV_INT(v_int64, VTraits<v_int64>::nlanes)

#define OPENCV_HAL_IMPL_RVV_BIN_OP_FP(_Tpvec, vl) \
inline _Tpvec v_add(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vfadd(a, b, vl); \
} \
inline _Tpvec v_sub(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vfsub(a, b, vl); \
} \
inline _Tpvec v_mul(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vfmul(a, b, vl); \
} \
inline _Tpvec v_div(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vfdiv(a, b, vl); \
} \
template<typename... Args> \
inline _Tpvec v_add(_Tpvec f1, _Tpvec f2, Args... vf) { \
    return v_add(vfadd(f1, f2, VTraits<_Tpvec>::nlanes), vf...); \
} \
template<typename... Args> \
inline _Tpvec v_mul(_Tpvec f1, _Tpvec f2, Args... vf) { \
    return v_mul(vfmul(f1, f2, VTraits<_Tpvec>::nlanes), vf...); \
}

OPENCV_HAL_IMPL_RVV_BIN_OP_FP( v_float32, VTraits<v_float32>::nlanes)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_BIN_OP_FP( v_float64, VTraits<v_float64>::nlanes)
#endif

////////////// Bitwise logic //////////////

#define OPENCV_HAL_IMPL_RVV_LOGIC_OP(_Tpvec, vl) \
inline _Tpvec v_not (const _Tpvec& a) \
{ \
    return vnot(a, vl); \
}

OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint8, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int8, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint16, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int16, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint32, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int32, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint64, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int64, VTraits<v_int64>::nlanes)

#define OPENCV_HAL_IMPL_RVV_FLT32_BIT_OP(op, vl) \
inline v_float32 v_##op (const v_float32& a, const v_float32& b) \
{ \
    return vreinterpret_v_i32m1_f32m1(v##op(vreinterpret_v_f32m1_i32m1(a), vreinterpret_v_f32m1_i32m1(b), vl)); \
}
OPENCV_HAL_IMPL_RVV_FLT32_BIT_OP(and, VTraits<v_float32>::nlanes)
OPENCV_HAL_IMPL_RVV_FLT32_BIT_OP(or, VTraits<v_float32>::nlanes)
OPENCV_HAL_IMPL_RVV_FLT32_BIT_OP(xor, VTraits<v_float32>::nlanes)

inline v_float32 v_not(const v_float32& a)
{
    return vreinterpret_v_i32m1_f32m1(vnot(vreinterpret_v_f32m1_i32m1(a), VTraits<v_float32>::nlanes));
}

#if CV_SIMD128_64F
#define OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(op, vl) \
inline v_float64 v_##op (const v_float64& a, const v_float64& b) \
{ \
    return vreinterpret_v_i64m1_f64m1(v##op(vreinterpret_v_f64m1_i64m1(a), vreinterpret_v_f64m1_i64m1(b), vl)); \
}
OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(and, VTraits<v_float64>::nlanes)
OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(or, VTraits<v_float64>::nlanes)
OPENCV_HAL_IMPL_RVV_FLT64_BIT_OP(xor, VTraits<v_float64>::nlanes)
inline v_float64 v_not (const v_float64& a)
{
    return vreinterpret_v_i64m1_f64m1(vnot(vreinterpret_v_f64m1_i64m1(a), VTraits<v_float64>::nlanes));
}
#endif

////////////// Bitwise shifts //////////////

#define OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(_Tpvec, vl) \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ \
    return _Tpvec(vsll(a, uint8_t(n), vl)); \
} \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ \
    return _Tpvec(vsrl(a, uint8_t(n), vl)); \
}

#define OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(_Tpvec, vl) \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ \
    return _Tpvec(vsll(a, uint8_t(n), vl)); \
} \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ \
    return _Tpvec(vsra(a, uint8_t(n), vl)); \
}

OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint8, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint16, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint32, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint64, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int8, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int16, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int32, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int64, VTraits<v_int64>::nlanes)

////////////// Comparison //////////////

#define OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, op, intrin, suffix, vl) \
inline _Tpvec v_##op(const _Tpvec& a, const _Tpvec& b) \
{ \
    uint64_t ones = -1; \
    return vmerge(intrin(a, b, vl), vmv_v_x_##suffix##m1(0, vl), ones, vl); \
}

#define OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, op, intrin, suffix, vl) \
inline _Tpvec v_##op (const _Tpvec& a, const _Tpvec& b) \
{ \
    union { uint64 u; double d; } ones; ones.u = -1; \
    return _Tpvec(vfmerge(intrin(a, b, vl), vfmv_v_f_##suffix##m1(0, vl), ones.d, vl)); \
} //TODO

#define OPENCV_HAL_IMPL_RVV_UNSIGNED_CMP(_Tpvec, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, eq, vmseq, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, ne, vmsne, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, lt, vmsltu, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, gt, vmsgtu, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, le, vmsleu, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, ge, vmsgeu, suffix, vl)

#define OPENCV_HAL_IMPL_RVV_SIGNED_CMP(_Tpvec, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, eq, vmseq, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, ne, vmsne, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, lt, vmslt, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, gt, vmsgt, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, le, vmsle, suffix, vl) \
OPENCV_HAL_IMPL_RVV_INT_CMP_OP(_Tpvec, ge, vmsge, suffix, vl)

#define OPENCV_HAL_IMPL_RVV_FLOAT_CMP(_Tpvec, suffix, vl) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, eq, vmfeq, suffix, vl) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, ne, vmfne, suffix, vl) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, lt, vmflt, suffix, vl) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, gt, vmfgt, suffix, vl) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, le, vmfle, suffix, vl) \
OPENCV_HAL_IMPL_RVV_FLOAT_CMP_OP(_Tpvec, ge, vmfge, suffix, vl)


OPENCV_HAL_IMPL_RVV_UNSIGNED_CMP(v_uint8, u8, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_UNSIGNED_CMP(v_uint16, u16, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_UNSIGNED_CMP(v_uint32, u32, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_UNSIGNED_CMP(v_uint64, u64, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_SIGNED_CMP(v_int8, i8, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_SIGNED_CMP(v_int16, i16, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_SIGNED_CMP(v_int32, i32, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_SIGNED_CMP(v_int64, i64, VTraits<v_int64>::nlanes)
OPENCV_HAL_IMPL_RVV_FLOAT_CMP(v_float32, f32, VTraits<v_float32>::nlanes)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_FLOAT_CMP(v_float64, f64, VTraits<v_float64>::nlanes)
#endif

inline v_float32 v_not_nan(const v_float32& a)
{ return v_eq(a, a); }

#if CV_SIMD128_64F
inline v_float64 v_not_nan(const v_float64& a)
{ return v_eq(a, a); }
#endif

////////////// Min/Max //////////////

#define OPENCV_HAL_IMPL_RVV_BIN_FUNC(_Tpvec, func, intrin, vl) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return intrin(a, b, vl); \
}

OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8, v_min, vminu, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8, v_max, vmaxu, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8, v_min, vmin, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8, v_max, vmax, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16, v_min, vminu, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16, v_max, vmaxu, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16, v_min, vmin, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16, v_max, vmax, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint32, v_min, vminu, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint32, v_max, vmaxu, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int32, v_min, vmin, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int32, v_max, vmax, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float32, v_min, vfmin, VTraits<v_float32>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float32, v_max, vfmax, VTraits<v_float32>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint64, v_min, vminu, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint64, v_max, vmaxu, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int64, v_min, vmin, VTraits<v_int64>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int64, v_max, vmax, VTraits<v_int64>::nlanes)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float64, v_min, vfmin, VTraits<v_float64>::nlanes)
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float64, v_max, vfmax, VTraits<v_float64>::nlanes)
#endif

////////////// Reduce //////////////

#define OPENCV_HAL_IMPL_RVV_REDUCE_SUM(_Tpvec, _wTpvec, _nwTpvec, scalartype, wsuffix, vl, red) \
inline scalartype v_reduce_sum(const _Tpvec& a)  \
{ \
    _nwTpvec zero = vmv_v_x_##wsuffix##m1(0, vl); \
    _nwTpvec res = vmv_v_x_##wsuffix##m1(0, vl); \
    res = v##red(res, a, zero, vl); \
    return (scalartype)v_get0(res); \
}

OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint8, v_uint16, vuint16m1_t, unsigned, u16, VTraits<v_uint8>::nlanes, wredsumu)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int8, v_int16, vint16m1_t, int, i16, VTraits<v_int8>::nlanes, wredsum)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint16, v_uint32, vuint32m1_t, unsigned, u32, VTraits<v_uint16>::nlanes, wredsumu)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int16, v_int32, vint32m1_t, int, i32, VTraits<v_int16>::nlanes, wredsum)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint32, v_uint64, vuint64m1_t, unsigned, u64, VTraits<v_uint32>::nlanes, wredsumu)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int32, v_int64, vint64m1_t, int, i64, VTraits<v_int32>::nlanes, wredsum)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint64, v_uint64, vuint64m1_t, uint64, u64, VTraits<v_uint64>::nlanes, redsum)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int64, v_int64, vint64m1_t, int64, i64, VTraits<v_int64>::nlanes, redsum)

#define OPENCV_HAL_IMPL_RVV_REDUCE_SUM_FP(_Tpvec, _wTpvec, _nwTpvec, scalartype, wsuffix, vl) \
inline scalartype v_reduce_sum(const _Tpvec& a)  \
{ \
    _nwTpvec zero = vfmv_v_f_##wsuffix##m1(0, vl); \
    _nwTpvec res = vfmv_v_f_##wsuffix##m1(0, vl); \
    res = vfredosum(res, a, zero, vl); \
    return (scalartype)v_get0(res); \
}

// vfredsum for float has renamed to vfredosum, also updated in GNU.
OPENCV_HAL_IMPL_RVV_REDUCE_SUM_FP(v_float32, v_float32, vfloat32m1_t, float, f32, VTraits<v_float32>::nlanes)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_REDUCE_SUM_FP(v_float64, v_float64, vfloat64m1_t, double, f64, VTraits<v_float64>::nlanes)
#endif


#define OPENCV_HAL_IMPL_RVV_REDUCE(_Tpvec, func, scalartype, suffix, vl, red) \
inline scalartype v_reduce_##func(const _Tpvec& a)  \
{ \
    _Tpvec res = _Tpvec(v##red(a, a, a, vl)); \
    return (scalartype)v_get0(res); \
}

OPENCV_HAL_IMPL_RVV_REDUCE(v_uint8, min, uchar, u8, VTraits<v_uint8>::nlanes, redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int8, min, schar, i8, VTraits<v_int8>::nlanes, redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint16, min, ushort, u16, VTraits<v_uint16>::nlanes, redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int16, min, short, i16, VTraits<v_int16>::nlanes, redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint32, min, unsigned, u32, VTraits<v_uint32>::nlanes, redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int32, min, int, i32, VTraits<v_int32>::nlanes, redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_float32, min, float, f32, VTraits<v_float32>::nlanes, fredmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint8, max, uchar, u8, VTraits<v_uint8>::nlanes, redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int8, max, schar, i8, VTraits<v_int8>::nlanes, redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint16, max, ushort, u16, VTraits<v_uint16>::nlanes, redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int16, max, short, i16, VTraits<v_int16>::nlanes, redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint32, max, unsigned, u32, VTraits<v_uint32>::nlanes, redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int32, max, int, i32, VTraits<v_int32>::nlanes, redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_float32, max, float, f32, VTraits<v_float32>::nlanes, fredmax)

inline v_float32 v_reduce_sum4(const v_float32& a, const v_float32& b,
                                 const v_float32& c, const v_float32& d)
{
    float elems[4] =
    {
        v_reduce_sum(a),
        v_reduce_sum(b),
        v_reduce_sum(c),
        v_reduce_sum(d)
    };
    return vle32_v_f32m1(elems, 4);
}

////////////// Square-Root //////////////

inline v_float32 v_sqrt(const v_float32& x)
{
    return vfsqrt(x, VTraits<v_float32>::nlanes);
}

inline v_float32 v_invsqrt(const v_float32& x)
{
    v_float32 one = v_setall_f32(1.0f);
    return v_div(one, v_sqrt(x));
}

#if CV_SIMD128_64F
inline v_float64 v_sqrt(const v_float64& x)
{
    return vfsqrt(x, VTraits<v_float64>::nlanes);
}

inline v_float64 v_invsqrt(const v_float64& x)
{
    v_float64 one = v_setall_f64(1.0f);
    return v_div(one, v_sqrt(x));
}
#endif

inline v_float32 v_magnitude(const v_float32& a, const v_float32& b)
{
    v_float32 x = vfmacc(vfmul(a, a, VTraits<v_float32>::nlanes), b, b, VTraits<v_float32>::nlanes);
    return v_sqrt(x);
}

inline v_float32 v_sqr_magnitude(const v_float32& a, const v_float32& b)
{
    return v_float32(vfmacc(vfmul(a, a, VTraits<v_float32>::nlanes), b, b, VTraits<v_float32>::nlanes));
}

#if CV_SIMD128_64F
inline v_float64 v_magnitude(const v_float64& a, const v_float64& b)
{
    v_float64 x = vfmacc(vfmul(a, a, VTraits<v_float64>::nlanes), b, b, VTraits<v_float64>::nlanes);
    return v_sqrt(x);
}

inline v_float64 v_sqr_magnitude(const v_float64& a, const v_float64& b)
{
    return vfmacc(vfmul(a, a, VTraits<v_float64>::nlanes), b, b, VTraits<v_float64>::nlanes);
}
#endif


////////////// Multiply-Add //////////////

inline v_float32 v_fma(const v_float32& a, const v_float32& b, const v_float32& c)
{
    return vfmacc(c, a, b, VTraits<v_float32>::nlanes);
}
inline v_int32 v_fma(const v_int32& a, const v_int32& b, const v_int32& c)
{
    return vmacc(c, a, b, VTraits<v_float32>::nlanes);
}

inline v_float32 v_muladd(const v_float32& a, const v_float32& b, const v_float32& c)
{
    return v_fma(a, b, c);
}

inline v_int32 v_muladd(const v_int32& a, const v_int32& b, const v_int32& c)
{
    return v_fma(a, b, c);
}

#if CV_SIMD128_64F
inline v_float64 v_fma(const v_float64& a, const v_float64& b, const v_float64& c)
{
    return vfmacc_vv_f64m1(c, a, b, VTraits<v_float64>::nlanes);
}

inline v_float64 v_muladd(const v_float64& a, const v_float64& b, const v_float64& c)
{
    return v_fma(a, b, c);
}
#endif

////////////// Check all/any //////////////

// use overloaded vcpop in clang, no casting like (vuint64m1_t) is needed.
#define OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(_Tpvec, vl) \
inline bool v_check_all(const _Tpvec& a) \
{ \
    return vcpop(vmslt(a, 0, vl), vl) == vl; \
} \
inline bool v_check_any(const _Tpvec& a) \
{ \
    return vcpop(vmslt(a, 0, vl), vl) != 0; \
}

OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int8, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int16, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int32, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int64, VTraits<v_int64>::nlanes)


inline bool v_check_all(const v_uint8& a)
{ return v_check_all(v_reinterpret_as_s8(a)); }
inline bool v_check_any(const v_uint8& a)
{ return v_check_any(v_reinterpret_as_s8(a)); }

inline bool v_check_all(const v_uint16& a)
{ return v_check_all(v_reinterpret_as_s16(a)); }
inline bool v_check_any(const v_uint16& a)
{ return v_check_any(v_reinterpret_as_s16(a)); }

inline bool v_check_all(const v_uint32& a)
{ return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_uint32& a)
{ return v_check_any(v_reinterpret_as_s32(a)); }

inline bool v_check_all(const v_float32& a)
{ return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_float32& a)
{ return v_check_any(v_reinterpret_as_s32(a)); }

inline bool v_check_all(const v_uint64& a)
{ return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_uint64& a)
{ return v_check_any(v_reinterpret_as_s64(a)); }

#if CV_SIMD128_64F
inline bool v_check_all(const v_float64& a)
{ return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_float64& a)
{ return v_check_any(v_reinterpret_as_s64(a)); }
#endif

////////////// abs //////////////

#define OPENCV_HAL_IMPL_RVV_ABSDIFF(_Tpvec, abs) \
inline _Tpvec v_##abs(const _Tpvec& a, const _Tpvec& b) \
{ \
    return v_sub(v_max(a, b), v_min(a, b)); \
}

OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint8, absdiff)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint16, absdiff)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint32, absdiff)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_float32, absdiff)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_float64, absdiff)
#endif
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int8, absdiffs)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int16, absdiffs)

// use reinterpret instead of c-style casting.
#define OPENCV_HAL_IMPL_RVV_ABSDIFF_S(_Tpvec, _rTpvec, width, vl) \
inline _rTpvec v_absdiff(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vnclipu(vreinterpret_u##width##m2(vwsub_vv(v_max(a, b), v_min(a, b), vl)), 0, vl); \
}

OPENCV_HAL_IMPL_RVV_ABSDIFF_S(v_int8, v_uint8, 16, 16)
OPENCV_HAL_IMPL_RVV_ABSDIFF_S(v_int16, v_uint16, 32, 8)
OPENCV_HAL_IMPL_RVV_ABSDIFF_S(v_int32, v_uint32, 64, 4)

#define OPENCV_HAL_IMPL_RVV_ABS(_Tprvec, _Tpvec, suffix) \
inline _Tprvec v_abs(const _Tpvec& a) \
{ \
    return v_absdiff(a, v_setzero_##suffix()); \
}

OPENCV_HAL_IMPL_RVV_ABS(v_uint8, v_int8, s8)
OPENCV_HAL_IMPL_RVV_ABS(v_uint16, v_int16, s16)
OPENCV_HAL_IMPL_RVV_ABS(v_uint32, v_int32, s32)
OPENCV_HAL_IMPL_RVV_ABS(v_float32, v_float32, f32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_ABS(v_float64, v_float64, f64)
#endif


#define OPENCV_HAL_IMPL_RVV_REDUCE_SAD(_Tpvec, scalartype) \
inline scalartype v_reduce_sad(const _Tpvec& a, const _Tpvec& b) \
{ \
    return v_reduce_sum(v_absdiff(a, b)); \
}

OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint8, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int8, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint16, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int16, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint32, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int32, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_float32, float)

////////////// Select //////////////

#define OPENCV_HAL_IMPL_RVV_SELECT(_Tpvec, vl) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return vmerge(vmsne(mask, 0, vl), b, a, vl); \
}

OPENCV_HAL_IMPL_RVV_SELECT(v_uint8, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_SELECT(v_uint16, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_SELECT(v_uint32, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_SELECT(v_int8, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_SELECT(v_int16, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_SELECT(v_int32, VTraits<v_int32>::nlanes)

inline v_float32 v_select(const v_float32& mask, const v_float32& a, const v_float32& b) \
{ \
    return vmerge(vmfne(mask, 0, VTraits<v_float32>::nlanes), b, a, VTraits<v_float32>::nlanes); \
}

#if CV_SIMD128_64F
inline v_float64 v_select(const v_float64& mask, const v_float64& a, const v_float64& b) \
{ \
    return vmerge(vmfne(mask, 0, VTraits<v_float64>::nlanes), b, a, VTraits<v_float64>::nlanes); \
}
#endif


////////////// Rotate shift //////////////

#define OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(_Tpvec, suffix, vl) \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{ \
    return vslidedown(vmv_v_x_##suffix##m1(0, vl), a, n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{ \
    return vslideup(vmv_v_x_##suffix##m1(0, vl), a, n, vl); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ return a; } \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(vslidedown(vmv_v_x_##suffix##m1(0, vl), a, n, vl), b, VTraits<_Tpvec>::nlanes - n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(vslidedown(vmv_v_x_##suffix##m1(0, vl), b, VTraits<_Tpvec>::nlanes - n, vl), a, n, vl); \
} \
// template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
// { CV_UNUSED(b); return a; }

OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint8, u8, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int8, i8, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint16, u16, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int16, i16,  VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint32, u32, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int32, i32, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint64, u64, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int64, i64, VTraits<v_int64>::nlanes)

#define OPENCV_HAL_IMPL_RVV_ROTATE_FP(_Tpvec, suffix, vl) \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{ \
    return vslidedown(vfmv_v_f_##suffix##m1(0, vl), a, n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{ \
    return vslideup(vfmv_v_f_##suffix##m1(0, vl), a, n, vl); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ return a; } \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(vslidedown(vfmv_v_f_##suffix##m1(0, vl), a, n, vl), b, VTraits<_Tpvec>::nlanes - n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(vslidedown(vfmv_v_f_##suffix##m1(0, vl), b, VTraits<_Tpvec>::nlanes - n, vl), a, n, vl); \
} \
// template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
// { CV_UNUSED(b); return a; }

OPENCV_HAL_IMPL_RVV_ROTATE_FP(v_float32, f32, VTraits<v_float32>::nlanes)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_ROTATE_FP(v_float64, f64,  VTraits<v_float64>::nlanes)
#endif

////////////// Convert to float //////////////

inline v_float32 v_cvt_f32(const v_int32& a)
{
    return vfcvt_f_x_v_f32m1(a, VTraits<v_float32>::nlanes);
}
// TODO: F64
#if CV_SIMD128_64F
inline v_float32 v_cvt_f32(const v_float64& a)
{
    return vfncvt_f(vmset_m_b32(VTraits<v_float32>::nlanes), v_setzero_f32(), vlmul_ext_f64m2(a), VTraits<v_float64>::nlanes);
}

inline v_float32 v_cvt_f32(const v_float64& a, const v_float64& b)
{
    return vfncvt_f(vset(vlmul_ext_f64m2(a),1,b), VTraits<v_float32>::nlanes);
}

inline v_float64 v_cvt_f64(const v_int32& a)
{
    return vget_f64m1(vfwcvt_f(a, VTraits<v_int32>::nlanes), 0);
}

inline v_float64 v_cvt_f64_high(const v_int32& a)
{
    return vget_f64m1(vfwcvt_f(a, VTraits<v_int32>::nlanes), 1);
}

inline v_float64 v_cvt_f64(const v_float32& a)
{
    return vget_f64m1(vfwcvt_f(a, VTraits<v_float32>::nlanes), 0);
}

inline v_float64 v_cvt_f64_high(const v_float32& a)
{
    return vget_f64m1(vfwcvt_f(a, VTraits<v_float32>::nlanes), 1);
}

inline v_float64 v_cvt_f64(const v_int64& a)
{
    return vfcvt_f(a, VTraits<v_int64>::nlanes);
}
#endif

//////////// Broadcast //////////////

#define OPENCV_HAL_IMPL_RVV_BROADCAST(_Tpvec, suffix) \
template<int s = 0> inline _Tpvec v_broadcast_element(_Tpvec v, int i = s) \
{ \
    return v_setall_##suffix(v_extract_n(v, i)); \
}

OPENCV_HAL_IMPL_RVV_BROADCAST(v_uint8, u8)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_int8, s8)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_uint16, u16)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_int16, s16)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_uint32, u32)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_int32, s32)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_uint64, u64)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_int64, s64)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_float32, f32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_BROADCAST(v_float64, f64)
#endif

////////////// Transpose4x4 //////////////

#define OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(_Tpvec, _Tp, suffix) \
inline void v_transpose4x4(const v_##_Tpvec& a0, const v_##_Tpvec& a1, \
                         const v_##_Tpvec& a2, const v_##_Tpvec& a3, \
                         v_##_Tpvec& b0, v_##_Tpvec& b1, \
                         v_##_Tpvec& b2, v_##_Tpvec& b3) \
{ \
    _Tp elems0[4] = \
    { \
        v_extract_n<0>(a0), \
        v_extract_n<0>(a1), \
        v_extract_n<0>(a2), \
        v_extract_n<0>(a3) \
    }; \
    b0 = vle32_v_##suffix##m1(elems0, 4); \
    _Tp elems1[4] = \
    { \
        v_extract_n<1>(a0), \
        v_extract_n<1>(a1), \
        v_extract_n<1>(a2), \
        v_extract_n<1>(a3) \
    }; \
    b1 = vle32_v_##suffix##m1(elems1, 4); \
    _Tp elems2[4] = \
    { \
        v_extract_n<2>(a0), \
        v_extract_n<2>(a1), \
        v_extract_n<2>(a2), \
        v_extract_n<2>(a3) \
    }; \
    b2 = vle32_v_##suffix##m1(elems2, 4); \
    _Tp elems3[4] = \
    { \
        v_extract_n<3>(a0), \
        v_extract_n<3>(a1), \
        v_extract_n<3>(a2), \
        v_extract_n<3>(a3) \
    }; \
    b3 = vle32_v_##suffix##m1(elems3, 4); \
}

OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(uint32, unsigned, u32)
OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(int32, int, i32)
OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(float32, float, f32)

////////////// Reverse //////////////

#define OPENCV_HAL_IMPL_RVV_REVERSE(_Tpvec, _Tp, suffix, idxType, width) \
inline _Tpvec v_reverse(const _Tpvec& a)  \
{ \
    idxType idx[] = {31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0}; \
    vuint##width##m1_t vidx = v_load(idx + 32 - VTraits<_Tpvec>::nlanes); \
    return vrgather(a, vidx, VTraits<_Tpvec>::nlanes); \
}

OPENCV_HAL_IMPL_RVV_REVERSE(v_uint8, uchar, u8, uchar, 8)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int8, schar, i8, uchar, 8)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint16, ushort, u16, ushort, 16)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int16, short, i16, ushort, 16)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint32, uint, u32, uint, 32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int32, int, i32, uint, 32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_float32, float, f32, uint, 32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint64, uint64, u64, uint64, 64)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int64, int64, i64, uint64, 64)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_REVERSE(v_float64, double, f64, uint64, 64)
#endif

//////////// Value reordering ////////////

#define OPENCV_HAL_IMPL_RVV_EXPAND(_Tp, _Tpwvec, _Tpwvec_m2, _Tpvec, width, suffix, suffix2, cvt) \
inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1) \
{ \
    _Tpwvec_m2 temp = cvt(a, vsetvlmax_e##width##m1()); \
    b0 = vget_##suffix##m1(temp, 0); \
    b1 = vget_##suffix##m1(temp, 1); \
} \
inline _Tpwvec v_expand_low(const _Tpvec& a) \
{ \
    _Tpwvec_m2 temp = cvt(a, vsetvlmax_e##width##m1()); \
    return vget_##suffix##m1(temp, 0); \
} \
inline _Tpwvec v_expand_high(const _Tpvec& a) \
{ \
    _Tpwvec_m2 temp = cvt(a, vsetvlmax_e##width##m1()); \
    return vget_##suffix##m1(temp, 1); \
} \
inline _Tpwvec v_load_expand(const _Tp* ptr) \
{ \
    return cvt(vle##width##_v_##suffix2##mf2(ptr, vsetvlmax_e##width##m1()), vsetvlmax_e##width##m1()); \
}

OPENCV_HAL_IMPL_RVV_EXPAND(uchar, v_uint16, vuint16m2_t, v_uint8, 8, u16, u8, vwcvtu_x)
OPENCV_HAL_IMPL_RVV_EXPAND(schar, v_int16, vint16m2_t, v_int8, 8, i16, i8, vwcvt_x)
OPENCV_HAL_IMPL_RVV_EXPAND(ushort, v_uint32, vuint32m2_t, v_uint16, 16, u32, u16, vwcvtu_x)
OPENCV_HAL_IMPL_RVV_EXPAND(short, v_int32, vint32m2_t, v_int16, 16, i32, i16, vwcvt_x)
OPENCV_HAL_IMPL_RVV_EXPAND(uint, v_uint64, vuint64m2_t, v_uint32, 32, u64, u32, vwcvtu_x)
OPENCV_HAL_IMPL_RVV_EXPAND(int, v_int64, vint64m2_t, v_int32, 32, i64, i32, vwcvt_x)

inline v_uint32 v_load_expand_q(const uchar* ptr)
{
    return vwcvtu_x(vwcvtu_x(vle8_v_u8mf4(ptr, VTraits<v_uint32>::nlanes), VTraits<v_uint32>::nlanes), VTraits<v_uint32>::nlanes);
}

inline v_int32 v_load_expand_q(const schar* ptr)
{
    return vwcvt_x(vwcvt_x(vle8_v_i8mf4(ptr, VTraits<v_int32>::nlanes), VTraits<v_int32>::nlanes), VTraits<v_int32>::nlanes);
}

#define OPENCV_HAL_IMPL_RVV_PACK(_Tpvec, _Tp, _wTpvec, hwidth, hsuffix, suffix, rshr, shr) \
inline _Tpvec v_pack(const _wTpvec& a, const _wTpvec& b) \
{ \
    return shr(vset(vlmul_ext_##suffix##m2(a), 1, b), 0, VTraits<_Tpvec>::nlanes); \
} \
inline void v_pack_store(_Tp* ptr, const _wTpvec& a) \
{ \
    vse##hwidth##_v_##hsuffix##mf2(ptr, shr(a, 0, VTraits<_Tpvec>::nlanes), VTraits<_wTpvec>::nlanes); \
} \
template<int n = 0> inline \
_Tpvec v_rshr_pack(const _wTpvec& a, const _wTpvec& b, int N = n) \
{ \
    return rshr(vset(vlmul_ext_##suffix##m2(a), 1, b), N, VTraits<_Tpvec>::nlanes); \
} \
template<int n = 0> inline \
void v_rshr_pack_store(_Tp* ptr, const _wTpvec& a, int N = n) \
{ \
    vse##hwidth##_v_##hsuffix##mf2(ptr, rshr(a, N, VTraits<_Tpvec>::nlanes), VTraits<_wTpvec>::nlanes); \
}

OPENCV_HAL_IMPL_RVV_PACK(v_uint8, uchar, v_uint16, 8, u8, u16, vnclipu, vnclipu)
OPENCV_HAL_IMPL_RVV_PACK(v_int8, schar, v_int16, 8,  i8, i16, vnclip, vnclip)
OPENCV_HAL_IMPL_RVV_PACK(v_uint16, ushort, v_uint32, 16, u16, u32, vnclipu, vnclipu)
OPENCV_HAL_IMPL_RVV_PACK(v_int16, short, v_int32, 16, i16, i32, vnclip, vnclip)
OPENCV_HAL_IMPL_RVV_PACK(v_uint32, unsigned, v_uint64, 32, u32, u64, vnclipu, vnsrl)
OPENCV_HAL_IMPL_RVV_PACK(v_int32, int, v_int64, 32, i32, i64, vnclip, vnsra)

#define OPENCV_HAL_IMPL_RVV_PACK_U(_Tpvec, _Tp, _wTpvec, _wTp, hwidth, width, hsuffix, suffix, rshr, cast, hvl, vl) \
inline _Tpvec v_pack_u(const _wTpvec& a, const _wTpvec& b) \
{ \
    return vnclipu(cast(vmax(vset(vlmul_ext_##suffix##m2(a), 1, b), 0, vl)), 0, vl); \
} \
inline void v_pack_u_store(_Tp* ptr, const _wTpvec& a) \
{ \
    vse##hwidth##_v_##hsuffix##mf2(ptr, vnclipu(vreinterpret_u##width##m1(vmax(a, 0, vl)), 0, vl), hvl); \
} \
template<int N = 0> inline \
_Tpvec v_rshr_pack_u(const _wTpvec& a, const _wTpvec& b, int n = N) \
{ \
    return vnclipu(cast(vmax(vset(vlmul_ext_##suffix##m2(a), 1, b), 0, vl)), n, vl); \
} \
template<int N = 0> inline \
void v_rshr_pack_u_store(_Tp* ptr, const _wTpvec& a, int n = N) \
{ \
    vse##hwidth##_v_##hsuffix##mf2(ptr, vnclipu(vreinterpret_u##width##m1(vmax(a, 0, vl)), n, vl), hvl); \
}

OPENCV_HAL_IMPL_RVV_PACK_U(v_uint8, uchar, v_int16, short, 8, 16, u8, i16, vnclipu_wx_u8m1, vreinterpret_v_i16m2_u16m2, VTraits<v_int16>::nlanes, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_PACK_U(v_uint16, ushort, v_int32, int, 16, 32, u16, i32, vnclipu_wx_u16m1, vreinterpret_v_i32m2_u32m2, VTraits<v_int32>::nlanes, VTraits<v_uint16>::nlanes)

/* void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1)
  a0 = {A1 A2 A3 A4}
  a1 = {B1 B2 B3 B4}
---------------
  {A1 B1 A2 B2} and {A3 B3 A4 B4}
*/
#define OPENCV_HAL_IMPL_RVV_UNPACKS(_Tpvec, width) \
inline void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1) \
{ \
    for (size_t i = 0; i < VTraits<_Tpvec>::nlanes / 2; ++i) { \
        b0 = vslideup(vslidedown(a0,a0,i,VTraits<_Tpvec>::nlanes), b0, 1, VTraits<_Tpvec>::nlanes); \
        b0 = vslideup(vslidedown(a1,a1,i,VTraits<_Tpvec>::nlanes), b0, 1, VTraits<_Tpvec>::nlanes); \
        b1 = vslideup(vslidedown(a0,a0,i+VTraits<_Tpvec>::nlanes/2,VTraits<_Tpvec>::nlanes), b1, 1, VTraits<_Tpvec>::nlanes); \
        b1 = vslideup(vslidedown(a1,a1,i+VTraits<_Tpvec>::nlanes/2,VTraits<_Tpvec>::nlanes), b1, 1, VTraits<_Tpvec>::nlanes); \
    } \
    b0 = v_reverse(b0); \
    b1 = v_reverse(b1); \
} \
inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(a, b, VTraits<_Tpvec>::nlanes/2, VTraits<_Tpvec>::nlanes);\
} \
inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(vmnot(vmset_m_b##width(VTraits<_Tpvec>::nlanes/2),VTraits<_Tpvec>::nlanes), vslidedown(a, a,VTraits<_Tpvec>::nlanes/2,VTraits<_Tpvec>::nlanes), b, 0, VTraits<_Tpvec>::nlanes); \
} \
inline void v_recombine(const _Tpvec& a, const _Tpvec& b, _Tpvec& c, _Tpvec& d) \
{ \
    c = v_combine_low(a, b); \
    d = v_combine_high(a, b); \
}

OPENCV_HAL_IMPL_RVV_UNPACKS(v_uint8, 8)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_int8, 8)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_uint16, 16)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_int16, 16)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_uint32, 32)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_int32, 32)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_float32, 32)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_UNPACKS(v_float64, 64)
#endif

namespace hal {

enum StoreMode
{
    STORE_UNALIGNED = 0,
    STORE_ALIGNED = 1,
    STORE_ALIGNED_NOCACHE = 2
};

}

#define OPENCV_HAL_IMPL_RVV_INTERLEAVED(_Tpvec, _Tp, suffix, width, hwidth, vl) \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b) \
{ \
    a = vlse##width##_v_##suffix##m1(ptr  , sizeof(_Tp)*2, VTraits<v_##_Tpvec>::nlanes); \
    b = vlse##width##_v_##suffix##m1(ptr+1, sizeof(_Tp)*2, VTraits<v_##_Tpvec>::nlanes); \
}\
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b, v_##_Tpvec& c) \
{ \
    a = vlse##width##_v_##suffix##m1(ptr  , sizeof(_Tp)*3, VTraits<v_##_Tpvec>::nlanes); \
    b = vlse##width##_v_##suffix##m1(ptr+1, sizeof(_Tp)*3, VTraits<v_##_Tpvec>::nlanes); \
    c = vlse##width##_v_##suffix##m1(ptr+2, sizeof(_Tp)*3, VTraits<v_##_Tpvec>::nlanes); \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec& a, v_##_Tpvec& b, \
                                v_##_Tpvec& c, v_##_Tpvec& d) \
{ \
    \
    a = vlse##width##_v_##suffix##m1(ptr  , sizeof(_Tp)*4, VTraits<v_##_Tpvec>::nlanes); \
    b = vlse##width##_v_##suffix##m1(ptr+1, sizeof(_Tp)*4, VTraits<v_##_Tpvec>::nlanes); \
    c = vlse##width##_v_##suffix##m1(ptr+2, sizeof(_Tp)*4, VTraits<v_##_Tpvec>::nlanes); \
    d = vlse##width##_v_##suffix##m1(ptr+3, sizeof(_Tp)*4, VTraits<v_##_Tpvec>::nlanes); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    vsse##width(ptr, sizeof(_Tp)*2, a, VTraits<v_##_Tpvec>::nlanes); \
    vsse##width(ptr+1, sizeof(_Tp)*2, b, VTraits<v_##_Tpvec>::nlanes); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                const v_##_Tpvec& c, hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    vsse##width(ptr, sizeof(_Tp)*3, a, VTraits<v_##_Tpvec>::nlanes); \
    vsse##width(ptr+1, sizeof(_Tp)*3, b, VTraits<v_##_Tpvec>::nlanes); \
    vsse##width(ptr+2, sizeof(_Tp)*3, c, VTraits<v_##_Tpvec>::nlanes); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec& a, const v_##_Tpvec& b, \
                                const v_##_Tpvec& c, const v_##_Tpvec& d, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    vsse##width(ptr, sizeof(_Tp)*4, a, VTraits<v_##_Tpvec>::nlanes); \
    vsse##width(ptr+1, sizeof(_Tp)*4, b, VTraits<v_##_Tpvec>::nlanes); \
    vsse##width(ptr+2, sizeof(_Tp)*4, c, VTraits<v_##_Tpvec>::nlanes); \
    vsse##width(ptr+3, sizeof(_Tp)*4, d, VTraits<v_##_Tpvec>::nlanes); \
} \
inline v_##_Tpvec v_interleave_pairs(const v_##_Tpvec& vec) \
{ \
    _Tp ptr[RVV_VLEN_MAX/sizeof(_Tp)] = {0}; \
    _Tp ptrvec[RVV_VLEN_MAX/sizeof(_Tp)] = {0}; \
    v_store(ptrvec, vec); \
    for (int i = 0; i < VTraits<v_##_Tpvec>::nlanes/4; i++) \
    { \
        ptr[4*i  ] = ptrvec[4*i  ]; \
        ptr[4*i+1] = ptrvec[4*i+2]; \
        ptr[4*i+2] = ptrvec[4*i+1]; \
        ptr[4*i+3] = ptrvec[4*i+3]; \
    } \
    return v_load(ptr); \
} \
 inline v_##_Tpvec v_interleave_quads(const v_##_Tpvec& vec) \
{ \
    _Tp ptr[RVV_VLEN_MAX/sizeof(_Tp)] = {0}; \
    _Tp ptrvec[RVV_VLEN_MAX/sizeof(_Tp)] = {0}; \
    v_store(ptrvec, vec); \
    for (int i = 0; i < VTraits<v_##_Tpvec>::nlanes/8; i++) \
    { \
        ptr[8*i  ] = ptrvec[8*i  ]; \
        ptr[8*i+1] = ptrvec[8*i+4]; \
        ptr[8*i+2] = ptrvec[8*i+1]; \
        ptr[8*i+3] = ptrvec[8*i+5]; \
        ptr[8*i+4] = ptrvec[8*i+2]; \
        ptr[8*i+5] = ptrvec[8*i+6]; \
        ptr[8*i+6] = ptrvec[8*i+3]; \
        ptr[8*i+7] = ptrvec[8*i+7]; \
    } \
    return v_load(ptr); \
}

OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint8, uchar, u8, 8, 4, VTraits<v_uint8>::nlanes)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int8, schar, i8, 8, 4, VTraits<v_int8>::nlanes)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint16, ushort, u16, 16, 8, VTraits<v_uint16>::nlanes)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int16, short, i16, 16, 8, VTraits<v_int16>::nlanes)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint32, unsigned, u32, 32, 16, VTraits<v_uint32>::nlanes)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int32, int, i32, 32, 16, VTraits<v_int32>::nlanes)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(float32, float, f32, 32, 16, VTraits<v_float32>::nlanes)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(uint64, uint64, u64, 64, 32, VTraits<v_uint64>::nlanes)
OPENCV_HAL_IMPL_RVV_INTERLEAVED(int64, int64, i64, 64, 32, VTraits<v_int64>::nlanes)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_INTERLEAVED(float64, double, f64, 64, 32, VTraits<v_float64>::nlanes)
#endif

//////////// SignMask ////////////
#define OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(_Tpvec) \
inline int v_signmask(const _Tpvec& a) \
{ \
    uint8_t ans[4] = {0}; \
    vsm(ans, vmslt(a, 0, VTraits<_Tpvec>::nlanes), VTraits<_Tpvec>::nlanes); \
    return *(reinterpret_cast<int*>(ans)); \
} \
inline int v_scan_forward(const _Tpvec& a) \
{ \
    return (int)vfirst(vmslt(a, 0, VTraits<_Tpvec>::nlanes), VTraits<_Tpvec>::nlanes); \
} \


OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int8)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int16)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int32)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int64)
#if CV_SIMD128_64F
// OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_float64)
#endif

inline int v_signmask(const v_uint8& a)
{ return v_signmask(v_reinterpret_as_s8(a)); }
inline int v_signmask(const v_uint16& a)
{ return v_signmask(v_reinterpret_as_s16(a)); }
inline int v_signmask(const v_uint32& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }
inline int v_signmask(const v_float32& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }
inline int v_signmask(const v_uint64& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }
#if CV_SIMD128_64F
inline int v_signmask(const v_float64& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }
#endif

inline int v_scan_forward(const v_uint8& a)
{ return v_scan_forward(v_reinterpret_as_s8(a)); }
inline int v_scan_forward(const v_uint16& a)
{ return v_scan_forward(v_reinterpret_as_s16(a)); }
inline int v_scan_forward(const v_uint32& a)
{ return v_scan_forward(v_reinterpret_as_s32(a)); }
inline int v_scan_forward(const v_float32& a)
{ return v_scan_forward(v_reinterpret_as_s32(a)); }
inline int v_scan_forward(const v_uint64& a)
{ return v_scan_forward(v_reinterpret_as_s64(a)); }
#if CV_SIMD128_64F
inline int v_scan_forward(const v_float64& a)
{ return v_scan_forward(v_reinterpret_as_s64(a)); }
#endif

//////////// Pack triplets ////////////
// {A0, A1, A2, A3, B0, B1, B2, B3, C0 ...} --> {A0, A1, A2, B0, B1, B2, C0 ...}
// mask: {0,0,0,1, ...} -> {T,T,T,F, ...}
#define OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(_Tpvec, width) \
inline _Tpvec v_pack_triplets(const _Tpvec& vec) { \
    size_t vl = vsetvlmax_e8m1(); \
    vuint32m1_t one = vmv_v_x_u32m1(1, vl/4); \
    vuint8m1_t zero = vmv_v_x_u8m1(0, vl); \
    vuint8m1_t mask = vreinterpret_u8m1(one); \
    uint8_t tempMask[RVV_VLEN_MAX/8/8] = {0}; \
    vsm(tempMask, vmseq(vslideup(zero, mask, 3, vl), 0, vl), vl);\
    return vcompress(vlm_v_b##width(tempMask, VTraits<_Tpvec>::nlanes), vec, vec, VTraits<_Tpvec>::nlanes); \
}
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint8, 8)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int8, 8)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint16, 16)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int16, 16)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint32, 32)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int32, 32)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_float32, 32)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_uint64, 64)
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_int64, 64)
#if CV_SIMD128_64F
OPENCV_HAL_IMPL_RVV_PACK_TRIPLETS(v_float64, 64)
#endif

////// FP16 support ///////
/*
#if CV_FP16
inline v_float32 v_load_expand(const float16_t* ptr)
{
    return vfwcvt_f_f_v_f32m1(vle16_v_f16mf2(ptr, VTraits<v_float32>::nlanes), VTraits<v_float32>::nlanes);
}

inline void v_pack_store(float16_t* ptr, const v_float32& v)
{
    vse16_v_f16mf2(ptr, vfncvt_f_f_w_f16mf2(v, VTraits<v_float32>::nlanes), VTraits<v_float32>::nlanes);
}
#else
inline v_float32 v_load_expand(const float16_t* ptr)
{
    //TODO: VLEN=128 only
    const int N = 4;
    float buf[N];
    for( int i = 0; i < N; i++ ) buf[i] = (float)ptr[i];
    return v_load(buf);
}

inline void v_pack_store(float16_t* ptr, const v_float32& v)
{
    //TODO: VLEN=128 only
    const int N = 4; 
    float buf[N];
    v_store(buf, v);
    for( int i = 0; i < N; i++ ) ptr[i] = float16_t(buf[i]);
}
#endif
*/
////////////// Rounding //////////////
inline v_int32 v_round(const v_float32& a)
{
    return vfcvt_x(vfadd(a, 1e-6, VTraits<v_float32>::nlanes), VTraits<v_float32>::nlanes);
}

inline v_int32 v_floor(const v_float32& a)
{
    return vfcvt_x(vfsub(a, 0.5f - 1e-6, VTraits<v_float32>::nlanes), VTraits<v_float32>::nlanes);
}

inline v_int32 v_ceil(const v_float32& a)
{
    return vfcvt_x(vfadd(a, 0.5f - 1e-6, VTraits<v_float32>::nlanes), VTraits<v_float32>::nlanes);
}

inline v_int32 v_trunc(const v_float32& a)
{
    return vfcvt_rtz_x(a, VTraits<v_float32>::nlanes);
}
#if CV_SIMD128_64F
inline v_int32 v_round(const v_float64& a)
{
    return vfncvt_x(vlmul_ext_f64m2(vfadd(a, 1e-6, VTraits<v_float64>::nlanes)), VTraits<v_float32>::nlanes);
}

inline v_int32 v_round(const v_float64& a, const v_float64& b)
{
    return vfncvt_x(vset(vlmul_ext_f64m2(vfadd(a, 1e-6, VTraits<v_float64>::nlanes)), 1, b), VTraits<v_float32>::nlanes);
}

inline v_int32 v_floor(const v_float64& a)
{
    return vfncvt_x(vlmul_ext_f64m2(vfsub(a, 0.5f - 1e-6, VTraits<v_float64>::nlanes)), VTraits<v_float32>::nlanes);
}

inline v_int32 v_ceil(const v_float64& a)
{
    return vfncvt_x(vlmul_ext_f64m2(vfadd(a, 0.5f - 1e-6, VTraits<v_float64>::nlanes)), VTraits<v_float32>::nlanes);
}

inline v_int32 v_trunc(const v_float64& a)
{
    return vfncvt_rtz_x(vlmul_ext_f64m2(a), VTraits<v_float32>::nlanes);
}
#endif
static inline v_int32 v_interleave_add(const vint32m2_t& a) {
    vuint32m1_t one = vmv_v_x_u32m1(1, VTraits<v_uint32>::nlanes); \
    vuint16m1_t mask = vreinterpret_u16m1(one); \
    vbool16_t m = vmseq(mask, 1, VTraits<v_uint16>::nlanes); \
    vint32m2_t t1 = vcompress(m, a, a, VTraits<v_int32>::nlanes*2); \
    vint32m2_t t2 = vcompress(vmnot(m, VTraits<v_uint16>::nlanes), a, a, VTraits<v_int32>::nlanes*2); \
    return vlmul_trunc_i32m1(vadd(t1 ,t2, VTraits<v_int32>::nlanes)); \
}
static inline v_uint32 v_interleave_add(const vuint32m2_t& a) {
    vuint32m1_t one = vmv_v_x_u32m1(1, VTraits<v_uint32>::nlanes); \
    vuint16m1_t mask = vreinterpret_u16m1(one); \
    vbool16_t m = vmseq(mask, 1, VTraits<v_uint16>::nlanes); \
    vuint32m2_t t1 = vcompress(m, a, a, VTraits<v_uint32>::nlanes*2); \
    vuint32m2_t t2 = vcompress(vmnot(m, VTraits<v_uint16>::nlanes), a, a, VTraits<v_uint32>::nlanes*2); \
    return vlmul_trunc_u32m1(vadd(t1 ,t2, VTraits<v_uint32>::nlanes)); \
}
static inline v_int64 v_interleave_add(const vint64m2_t& a) {
    vuint64m1_t one = vmv_v_x_u64m1(1, VTraits<v_uint64>::nlanes); \
    vuint32m1_t mask = vreinterpret_u32m1(one); \
    vbool32_t m = vmseq(mask, 1, VTraits<v_uint32>::nlanes); \
    vint64m2_t t1 = vcompress(m, a, a, VTraits<v_int64>::nlanes*2); \
    vint64m2_t t2 = vcompress(vmnot(m, VTraits<v_uint32>::nlanes), a, a, VTraits<v_int64>::nlanes*2); \
    return vlmul_trunc_i64m1(vadd(t1 ,t2, VTraits<v_int64>::nlanes)); \
}
static inline v_uint64 v_interleave_add(const vuint64m2_t& a) {
    vuint64m1_t one = vmv_v_x_u64m1(1, VTraits<v_uint64>::nlanes); \
    vuint32m1_t mask = vreinterpret_u32m1(one); \
    vbool32_t m = vmseq(mask, 1, VTraits<v_uint32>::nlanes); \
    vuint64m2_t t1 = vcompress(m, a, a, VTraits<v_uint64>::nlanes*2); \
    vuint64m2_t t2 = vcompress(vmnot(m, VTraits<v_uint32>::nlanes), a, a, VTraits<v_uint64>::nlanes*2); \
    return vlmul_trunc_u64m1(vadd(t1 ,t2, VTraits<v_uint64>::nlanes)); \
}
// static inline v_int32 v_interleave_add(const vint16m1_t& a) {
//     vuint32m1_t one = vmv_v_x_u32m1(1, VTraits<v_uint32>::nlanes); \
//     vuint16m1_t mask = vreinterpret_u16m1(one); \
//     vbool16_t m = vmseq(mask, 1, VTraits<v_uint16>::nlanes); \
//     vint16m1_t t1 = vcompress(m, a, a, VTraits<v_uint16>::nlanes); \
//     vint16m1_t t2 = vcompress(vmnot(m, VTraits<v_uint16>::nlanes), a, a, VTraits<v_uint16>::nlanes); \
//     return vlmul_trunc_i32m1(vwadd_vv(t1 ,t2, VTraits<v_int32>::nlanes)); \
// }
//////// Dot Product ////////

// 16 >> 32
inline v_int32 v_dotprod(const v_int16& a, const v_int16& b)
{
    vint32m2_t tempAns = vwmul_vv_i32m2(a, b, VTraits<v_int16>::nlanes); \
    return v_interleave_add(tempAns);
}

inline v_int32 v_dotprod(const v_int16& a, const v_int16& b, const v_int32& c)
{
    vint32m2_t tempAns = vwmul_vv_i32m2(a, b, VTraits<v_int16>::nlanes); \
    return vadd(v_interleave_add(tempAns), c, VTraits<v_int32>::nlanes); \
}

// 32 >> 64
inline v_int64 v_dotprod(const v_int32& a, const v_int32& b)
{
    vuint64m1_t one = vmv_v_x_u64m1(1, VTraits<v_uint64>::nlanes); \
    vuint32m1_t mask = vreinterpret_u32m1(one); \
    vbool32_t m = vmseq(mask, 1, VTraits<v_uint32>::nlanes); \
    vint64m2_t tempAns = vwmul_vv_i64m2(a, b, VTraits<v_int32>::nlanes); \
    vint64m2_t t1 = vcompress(m, tempAns, tempAns, VTraits<v_int64>::nlanes*2); \
    vint64m2_t t2 = vcompress(vmnot(m, VTraits<v_uint32>::nlanes), tempAns, tempAns, VTraits<v_int64>::nlanes*2); \
    return vlmul_trunc_i64m1(vadd(t1 ,t2, VTraits<v_int64>::nlanes)); \
}
inline v_int64 v_dotprod(const v_int32& a, const v_int32& b, const v_int64& c)
{
    vuint64m1_t one = vmv_v_x_u64m1(1, VTraits<v_uint64>::nlanes); \
    vuint32m1_t mask = vreinterpret_u32m1(one); \
    vbool32_t m = vmseq(mask, 1, VTraits<v_uint32>::nlanes); \
    vint64m2_t tempAns = vwmul_vv_i64m2(a, b, VTraits<v_int32>::nlanes); \
    vint64m2_t t1 = vcompress(m, tempAns, tempAns, VTraits<v_int64>::nlanes*2); \
    vint64m2_t t2 = vcompress(vmnot(m, VTraits<v_uint32>::nlanes), tempAns, tempAns, VTraits<v_int64>::nlanes*2); \
    return vadd(vlmul_trunc_i64m1(vadd(t1 ,t2, VTraits<v_int64>::nlanes)), c, VTraits<v_int64>::nlanes); \
}

// 8 >> 32
inline v_uint32 v_dotprod_expand(const v_uint8& a, const v_uint8& b)
{
    vuint16m1_t one = vmv_v_x_u16m1(1, VTraits<v_uint16>::nlanes); \
    vuint8m1_t mask = vreinterpret_u8m1(one); \
    vbool8_t m = vmseq(mask, 1, VTraits<v_uint8>::nlanes); \
    vuint16m2_t tempAns = vwmulu(a, b, VTraits<v_uint8>::nlanes); \
    vuint16m2_t t1 = vcompress(m, tempAns, tempAns, VTraits<v_uint16>::nlanes*2); \
    vuint16m2_t t2 = vcompress(vmnot(m, VTraits<v_uint8>::nlanes), tempAns, tempAns, VTraits<v_uint16>::nlanes*2);
    vuint16m1_t t3 = vlmul_trunc_u16m1(vadd(t1, t2, VTraits<v_uint16>::nlanes));
    return v_interleave_add(vzext_vf2(t3, VTraits<v_uint16>::nlanes));
}

inline v_uint32 v_dotprod_expand(const v_uint8& a, const v_uint8& b,
                                  const v_uint32& c)
{
    vuint16m1_t one = vmv_v_x_u16m1(1, VTraits<v_uint16>::nlanes); \
    vuint8m1_t mask = vreinterpret_u8m1(one); \
    vbool8_t m = vmseq(mask, 1, VTraits<v_uint8>::nlanes); \
    vuint16m2_t tempAns = vwmulu(a, b, VTraits<v_uint8>::nlanes); \
    vuint16m2_t t1 = vcompress(m, tempAns, tempAns, VTraits<v_uint16>::nlanes*2); \
    vuint16m2_t t2 = vcompress(vmnot(m, VTraits<v_uint8>::nlanes), tempAns, tempAns, VTraits<v_uint16>::nlanes*2);
    vuint16m1_t t3 = vlmul_trunc_u16m1(vadd(t1, t2, VTraits<v_uint16>::nlanes));
    return vadd(v_interleave_add(vzext_vf2(t3, VTraits<v_uint16>::nlanes)), c, VTraits<v_uint32>::nlanes);
}

inline v_int32 v_dotprod_expand(const v_int8& a, const v_int8& b)
{
    vuint16m1_t one = vmv_v_x_u16m1(1, VTraits<v_uint16>::nlanes); \
    vuint8m1_t mask = vreinterpret_u8m1(one); \
    vbool8_t m = vmseq(mask, 1, VTraits<v_uint8>::nlanes); \
    vint16m2_t tempAns = vwmul(a, b, VTraits<v_int8>::nlanes); \
    vint16m2_t t1 = vcompress(m, tempAns, tempAns, VTraits<v_int16>::nlanes*2); \
    vint16m2_t t2 = vcompress(vmnot(m, VTraits<v_uint8>::nlanes), tempAns, tempAns, VTraits<v_int16>::nlanes*2);
    vint16m1_t t3 = vlmul_trunc_i16m1(vadd(t1, t2, VTraits<v_int16>::nlanes));
    return v_interleave_add(vsext_vf2(t3, VTraits<v_int16>::nlanes));
}

inline v_int32 v_dotprod_expand(const v_int8& a, const v_int8& b,
                                  const v_int32& c)
{
    vuint16m1_t one = vmv_v_x_u16m1(1, VTraits<v_uint16>::nlanes); \
    vuint8m1_t mask = vreinterpret_u8m1(one); \
    vbool8_t m = vmseq(mask, 1, VTraits<v_uint8>::nlanes); \
    vint16m2_t tempAns = vwmul(a, b, VTraits<v_int8>::nlanes); \
    vint16m2_t t1 = vcompress(m, tempAns, tempAns, VTraits<v_int16>::nlanes*2); \
    vint16m2_t t2 = vcompress(vmnot(m, VTraits<v_uint8>::nlanes), tempAns, tempAns, VTraits<v_int16>::nlanes*2);
    vint16m1_t t3 = vlmul_trunc_i16m1(vadd(t1, t2, VTraits<v_int16>::nlanes));
    return vadd(v_interleave_add(vsext_vf2(t3, VTraits<v_int16>::nlanes)), c, VTraits<v_int32>::nlanes);
}


// 16 >> 64
inline v_uint64 v_dotprod_expand(const v_uint16& a, const v_uint16& b)
{
    vuint32m1_t one = vmv_v_x_u32m1(1, VTraits<v_uint32>::nlanes); \
    vuint16m1_t mask = vreinterpret_u16m1(one); \
    vbool16_t m = vmseq(mask, 1, VTraits<v_uint16>::nlanes); \
    vuint32m2_t tempAns = vwmulu(a, b, VTraits<v_uint16>::nlanes); \
    vuint32m2_t t1 = vcompress(m, tempAns, tempAns, VTraits<v_uint32>::nlanes*2); \
    vuint32m2_t t2 = vcompress(vmnot(m, VTraits<v_uint16>::nlanes), tempAns, tempAns, VTraits<v_uint32>::nlanes*2);
    vuint32m1_t t3 = vlmul_trunc_u32m1(vadd(t1, t2, VTraits<v_uint32>::nlanes));
    return v_interleave_add(vzext_vf2(t3, VTraits<v_uint32>::nlanes));
}
inline v_uint64 v_dotprod_expand(const v_uint16& a, const v_uint16& b, const v_uint64& c)
{
    vuint32m1_t one = vmv_v_x_u32m1(1, VTraits<v_uint32>::nlanes); \
    vuint16m1_t mask = vreinterpret_u16m1(one); \
    vbool16_t m = vmseq(mask, 1, VTraits<v_uint16>::nlanes); \
    vuint32m2_t tempAns = vwmulu(a, b, VTraits<v_uint16>::nlanes); \
    vuint32m2_t t1 = vcompress(m, tempAns, tempAns, VTraits<v_uint32>::nlanes*2); \
    vuint32m2_t t2 = vcompress(vmnot(m, VTraits<v_uint16>::nlanes), tempAns, tempAns, VTraits<v_uint32>::nlanes*2);
    vuint32m1_t t3 = vlmul_trunc_u32m1(vadd(t1, t2, VTraits<v_uint32>::nlanes));
    return vadd(v_interleave_add(vzext_vf2(t3, VTraits<v_uint32>::nlanes)), c, VTraits<v_uint64>::nlanes);
}

inline v_int64 v_dotprod_expand(const v_int16& a, const v_int16& b)
{
    vuint32m1_t one = vmv_v_x_u32m1(1, VTraits<v_uint32>::nlanes); \
    vuint16m1_t mask = vreinterpret_u16m1(one); \
    vbool16_t m = vmseq(mask, 1, VTraits<v_uint16>::nlanes); \
    vint32m2_t tempAns = vwmul(a, b, VTraits<v_uint16>::nlanes); \
    vint32m2_t t1 = vcompress(m, tempAns, tempAns, VTraits<v_int32>::nlanes*2); \
    vint32m2_t t2 = vcompress(vmnot(m, VTraits<v_uint16>::nlanes), tempAns, tempAns, VTraits<v_int32>::nlanes*2);
    vint32m1_t t3 = vlmul_trunc_i32m1(vadd(t1, t2, VTraits<v_int32>::nlanes));
    return v_interleave_add(vsext_vf2(t3, VTraits<v_int32>::nlanes));
}
inline v_int64 v_dotprod_expand(const v_int16& a, const v_int16& b,
                                  const v_int64& c)
{
    vuint32m1_t one = vmv_v_x_u32m1(1, VTraits<v_uint32>::nlanes); \
    vuint16m1_t mask = vreinterpret_u16m1(one); \
    vbool16_t m = vmseq(mask, 1, VTraits<v_uint16>::nlanes); \
    vint32m2_t tempAns = vwmul(a, b, VTraits<v_uint16>::nlanes); \
    vint32m2_t t1 = vcompress(m, tempAns, tempAns, VTraits<v_int32>::nlanes*2); \
    vint32m2_t t2 = vcompress(vmnot(m, VTraits<v_uint16>::nlanes), tempAns, tempAns, VTraits<v_int32>::nlanes*2);
    vint32m1_t t3 = vlmul_trunc_i32m1(vadd(t1, t2, VTraits<v_int32>::nlanes));
    return vadd(v_interleave_add(vsext_vf2(t3, VTraits<v_int32>::nlanes)), c, VTraits<v_int64>::nlanes);
}

// 32 >> 64f
#if CV_SIMD128_64F
inline v_float64 v_dotprod_expand(const v_int32& a, const v_int32& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64 v_dotprod_expand(const v_int32& a,   const v_int32& b,
                                    const v_float64& c)
{ return v_add(v_dotprod_expand(a, b) , c); }
#endif

//////// Fast Dot Product ////////

// 16 >> 32
inline v_int32 v_dotprod_fast(const v_int16& a, const v_int16& b)
{
    v_int32 zero = v_setzero_s32();
    return vredsum(zero, vwmul(a, b, VTraits<v_int16>::nlanes), zero,  VTraits<v_int16>::nlanes);
}
inline v_int32 v_dotprod_fast(const v_int16& a, const v_int16& b, const v_int32& c)
{
    v_int32 zero = v_setzero_s32();
    return vredsum(zero, vwmul(a, b, VTraits<v_int16>::nlanes), c,  VTraits<v_int16>::nlanes);
}

// 32 >> 64
inline v_int64 v_dotprod_fast(const v_int32& a, const v_int32& b)
{
    v_int64 zero = v_setzero_s64();
    return vredsum(zero, vwmul(a, b, VTraits<v_int32>::nlanes), zero,  VTraits<v_int32>::nlanes);
}
inline v_int64 v_dotprod_fast(const v_int32& a, const v_int32& b, const v_int64& c)
{
    v_int64 zero = v_setzero_s64();
    return vadd(vredsum(zero, vwmul(a, b, VTraits<v_int32>::nlanes), zero,  VTraits<v_int32>::nlanes) , vredsum(zero, c, zero, VTraits<v_int64>::nlanes), VTraits<v_int64>::nlanes);
}


// 8 >> 32
inline v_uint32 v_dotprod_expand_fast(const v_uint8& a, const v_uint8& b)
{
    v_uint32 zero = v_setzero_u32();
    return vwredsumu(zero, vwmulu(a, b, VTraits<v_uint8>::nlanes), zero,  VTraits<v_uint8>::nlanes);
}
inline v_uint32 v_dotprod_expand_fast(const v_uint8& a, const v_uint8& b, const v_uint32& c)
{
    v_uint32 zero = v_setzero_u32();
    return vadd(vwredsumu(zero, vwmulu(a, b, VTraits<v_uint8>::nlanes), zero,  VTraits<v_uint8>::nlanes) , vredsum(zero, c, zero, VTraits<v_uint32>::nlanes), VTraits<v_uint32>::nlanes);
}
inline v_int32 v_dotprod_expand_fast(const v_int8& a, const v_int8& b)
{
    v_int32 zero = v_setzero_s32();
    return vwredsum(zero, vwmul(a, b, VTraits<v_int8>::nlanes), zero,  VTraits<v_int8>::nlanes);
}
inline v_int32 v_dotprod_expand_fast(const v_int8& a, const v_int8& b, const v_int32& c)
{
    v_int32 zero = v_setzero_s32();
    return vadd(vwredsum(zero, vwmul(a, b, VTraits<v_int8>::nlanes), zero,  VTraits<v_int8>::nlanes) , vredsum(zero, c, zero, VTraits<v_int32>::nlanes), VTraits<v_int32>::nlanes);
}

// 16 >> 64
inline v_uint64 v_dotprod_expand_fast(const v_uint16& a, const v_uint16& b)
{
    v_uint64 zero = v_setzero_u64();
    return vwredsumu(zero, vwmulu(a, b, VTraits<v_uint16>::nlanes), zero,  VTraits<v_uint16>::nlanes);
}
inline v_uint64 v_dotprod_expand_fast(const v_uint16& a, const v_uint16& b, const v_uint64& c)
{
    v_uint64 zero = v_setzero_u64();
    return vadd(vwredsumu(zero, vwmulu(a, b, VTraits<v_uint16>::nlanes), zero,  VTraits<v_uint16>::nlanes), vredsum(zero, c, zero, VTraits<v_uint64>::nlanes), VTraits<v_uint64>::nlanes);
}
inline v_int64 v_dotprod_expand_fast(const v_int16& a, const v_int16& b)
{
    v_int64 zero = v_setzero_s64();
    return vwredsum(zero, vwmul(a, b, VTraits<v_int16>::nlanes), zero,  VTraits<v_int16>::nlanes);
}
inline v_int64 v_dotprod_expand_fast(const v_int16& a, const v_int16& b, const v_int64& c)
{
    v_int64 zero = v_setzero_s64();
    return vadd(vwredsum(zero, vwmul(a, b, VTraits<v_int16>::nlanes), zero,  VTraits<v_int16>::nlanes), vredsum(zero, c, zero, VTraits<v_int64>::nlanes), VTraits<v_int64>::nlanes);
}

// 32 >> 64f
#if CV_SIMD128_64F
inline v_float64 v_dotprod_expand_fast(const v_int32& a, const v_int32& b)
{ return v_cvt_f64(v_dotprod_fast(a, b)); }
inline v_float64 v_dotprod_expand_fast(const v_int32& a, const v_int32& b, const v_float64& c)
{ return v_add(v_dotprod_expand_fast(a, b) , c); }
#endif

// WARNNING: 128 bit
inline v_float32 v_matmul(const v_float32& v, const v_float32& m0,
                            const v_float32& m1, const v_float32& m2,
                            const v_float32& m3)
{
    vfloat32m1_t res;
    res = vfmul_vf_f32m1(m0, v_extract_n(v, 0), 4);
    res = vfmacc_vf_f32m1(res, v_extract_n(v, 1), m1, 4);
    res = vfmacc_vf_f32m1(res, v_extract_n(v, 2), m2, 4);
    res = vfmacc_vf_f32m1(res, v_extract_n(v, 3), m3, 4);
    return res;
}

// TODO: test
inline v_float32 v_matmuladd(const v_float32& v, const v_float32& m0,
                               const v_float32& m1, const v_float32& m2,
                               const v_float32& a)
{
    vfloat32m1_t res = vfmul_vf_f32m1(m0, v_extract_n(v,0), 4);
    res = vfmacc_vf_f32m1(res, v_extract_n(v,1), m1, 4);
    res = vfmacc_vf_f32m1(res, v_extract_n(v,2), m2, 4);
    return vfadd(res, a, 4);
}

#define OPENCV_HAL_IMPL_RVV_MUL_EXPAND(_Tpvec, _Tpwvec, _TpwvecM2, suffix, wmul) \
inline void v_mul_expand(const _Tpvec& a, const _Tpvec& b, _Tpwvec& c, _Tpwvec& d) \
{ \
    _TpwvecM2 temp = wmul(a, b, VTraits<_Tpvec>::nlanes); \
    c = vget_##suffix##m1(temp, 0); \
    d = vget_##suffix##m1(temp, 1); \
}

OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint8, v_uint16, vuint16m2_t, u16, vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_int8, v_int16, vint16m2_t, i16, vwmul)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint16, v_uint32, vuint32m2_t, u32, vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_int16, v_int32, vint32m2_t, i32, vwmul)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint32, v_uint64, vuint64m2_t, u64, vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_int32, v_int64, vint64m2_t, i64, vwmul)


inline v_int16 v_mul_hi(const v_int16& a, const v_int16& b)
{
    return vmulh(a, b, VTraits<v_int16>::nlanes);
}
inline v_uint16 v_mul_hi(const v_uint16& a, const v_uint16& b)
{
    return vmulhu(a, b, VTraits<v_uint16>::nlanes);
}

//////// Saturating Multiply ////////

#define OPENCV_HAL_IMPL_RVV_MUL_SAT(_Tpvec, _wTpvec) \
inline _Tpvec v_mul(const _Tpvec& a, const _Tpvec& b) \
{ \
    _wTpvec c, d; \
    v_mul_expand(a, b, c, d); \
    return v_pack(c, d); \
} \

OPENCV_HAL_IMPL_RVV_MUL_SAT(v_uint8, v_uint16)
OPENCV_HAL_IMPL_RVV_MUL_SAT(v_int8, v_int16)
OPENCV_HAL_IMPL_RVV_MUL_SAT(v_uint16, v_uint32)
OPENCV_HAL_IMPL_RVV_MUL_SAT(v_int16, v_int32)

inline void v_cleanup() {}
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END
} //namespace rvv
} //namespace cv

#endif