#include "perf_precomp.hpp"
#include "../src/precomp.hpp"

namespace opencv_test
{
using namespace perf;

#if CV_SIMD128_CPP
typedef v_float64x2 v_float64;
#endif

typedef perf::TestBaseWithParam<size_t> Intrinsic;

#define VSTORE(_Tptr, _Tvec) \
inline void v_store_(void* ptr, const _Tvec& a) { v_store(reinterpret_cast<_Tptr>(ptr), a); }

VSTORE(uchar* ,v_uint8)
VSTORE(schar* ,v_int8)
VSTORE(ushort* ,v_uint16)
VSTORE(short* ,v_int16)
VSTORE(unsigned int* ,v_uint32)
VSTORE(int* ,v_int32)
VSTORE(uint64* ,v_uint64)
VSTORE(int64* ,v_int64)
VSTORE(float* ,v_float32)
VSTORE(double* ,v_float64)

#define CASES 1024,656636,4194304

#define OPENCV_HAL_PERF_TEST_LOAD(_Tpvec, _Tp, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> x(length); \
    vector<double> angle(length); \
    declare.in(x, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        startTimer(); \
        _Tpvec va = _Func(reinterpret_cast<_Tp *>(x.data())+i); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), va); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_ARG1I1O(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> x(length); \
    vector<double> angle(length); \
    declare.in(x, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(x.data())+i); \
        startTimer(); \
        auto vd = _Func(va); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vd); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_REDUCE(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> x(length); \
    vector<double> angle(length) ; \
    declare.in(x, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(x.data())+i); \
        startTimer(); \
        auto vd = _Func(va); \
        stopTimer(); \
        angle[i] = (double)vd; \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_REDUCE_SAD(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> x(length); \
    vector<double> y(length); \
    vector<double> angle(length) ; \
    declare.in(x, y, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(x.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(y.data())+i); \
        startTimer(); \
        auto vd = _Func(va, vb); \
        stopTimer(); \
        angle[i] = (double)vd; \
    } \
    SANITY_CHECK_NOTHING(); \
}

typedef tuple<int, int> Intrinsic_SHIFT_t;
typedef perf::TestBaseWithParam<Intrinsic_SHIFT_t> Intrinsic_SHIFT;

#define OPENCV_HAL_PERF_TEST_SHIFT(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic_SHIFT, _Func##_##_Tpvec, testing::Combine( testing::Values(CASES), testing::Values(0,1,2,3,4,5,6,7,8) )) \
{ \
    const int length = testing::get<0>(GetParam()); \
    const int offset = testing::get<1>(GetParam()); \
    vector<double> x(length); \
    vector<double> angle(length); \
    declare.in(x, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(x.data())+i); \
        startTimer(); \
        auto vd = _Func(va, offset); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vd); \
    } \
    SANITY_CHECK_NOTHING(); \
}
#define OPENCV_HAL_PERF_TEST_ROTATE(_Tpvec, _Func) \
_OPENCV_HAL_PERF_TEST_ROTATE(_Tpvec, _Func, 1)

#define _OPENCV_HAL_PERF_TEST_ROTATE(_Tpvec, _Func, _offset) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec##_offset, testing::Values(CASES) ) \
{ \
    const int length = GetParam(); \
    vector<double> x(length); \
    vector<double> angle(length); \
    declare.in(x, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(x.data())+i); \
        startTimer(); \
        auto vd = _Func<_offset>(va); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vd); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_LUT(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> x(length); \
    vector<int> idx(length); \
    vector<double> angle(length); \
    declare.in(x, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        startTimer(); \
        auto vd = _Func(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(x.data()), idx.data()); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vd); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_LUT_V(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_v##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> x(length); \
    vector<int> idx(length); \
    v_int32 vidx = v_load(idx.data()); \
    vector<double> angle(length); \
    declare.in(x, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        startTimer(); \
        auto vd = _Func(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(x.data()), vidx); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vd); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_ARG2I1O(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##__##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> Y(length); \
    vector<double> angle(length); \
    declare.in(X, Y, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(Y.data())+i); \
        startTimer(); \
        auto vc = _Func(va, vb); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vc); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_ARG2I0O(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##__##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> Y(length); \
    vector<double> angle(length); \
    vector<double> angle1(length); \
    declare.in(X, Y, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(Y.data())+i); \
        startTimer(); \
        _Func(va, vb); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), va); \
        v_store_(reinterpret_cast<void *>(angle1.data()), vb); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_ARG3I1O(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##___##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> Y(length); \
    vector<double> Z(length); \
    vector<double> angle(length); \
    declare.in(X, Y, Z, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(Y.data())+i); \
        _Tpvec vc = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(Z.data())+i); \
        startTimer(); \
        auto vd = _Func(va, vb, vc); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vd); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_ARG4I1O(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> angle(length); \
    declare.in(X, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vc = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vd = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        startTimer(); \
        auto vo = _Func(va, vb, vc, vd); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vo); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_ARG4I0O(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> angle(length); \
    vector<double> angle1(length); \
    declare.in(X, WARMUP_RNG).out(angle, angle1); \
    _Tpvec vc, vd;\
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        startTimer(); \
        _Func(va, vb, vc, vd); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vc); \
        v_store_(reinterpret_cast<void *>(angle1.data()), vd); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_ARG3I0O(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> angle(length); \
    declare.in(X, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vc; \
        startTimer(); \
        _Func(va, vb, vc); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vc); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_EXPAND(_Tpwvec, _Tpvec) \
PERF_TEST_P(Intrinsic, v_expand##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> angle(length); \
    vector<double> angle1(length); \
    declare.in(X, WARMUP_RNG).out(angle, angle1); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpwvec vb; \
        _Tpwvec vc; \
        startTimer(); \
        v_expand(va, vb, vc); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vb); \
        v_store_(reinterpret_cast<void *>(angle1.data()), vc); \
    } \
    SANITY_CHECK_NOTHING(); \
}

OPENCV_HAL_PERF_TEST_EXPAND(v_uint16, v_uint8)
OPENCV_HAL_PERF_TEST_EXPAND(v_int16, v_int8)
OPENCV_HAL_PERF_TEST_EXPAND(v_uint32, v_uint16)
OPENCV_HAL_PERF_TEST_EXPAND(v_int32, v_int16)
OPENCV_HAL_PERF_TEST_EXPAND(v_uint64, v_uint32)
OPENCV_HAL_PERF_TEST_EXPAND(v_int64, v_int32)

#define OPENCV_HAL_PERF_TEST_ARG8I0O(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> angle(length); \
    vector<double> angle1(length); \
    vector<double> angle2(length); \
    vector<double> angle3(length); \
    declare.in(X, WARMUP_RNG).out(angle, angle1, angle2, angle3); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vc = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vd = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec ve, vf, vg, vh; \
        startTimer(); \
        _Func(va, vb, vc, vd, ve, vf, vg, vh); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), ve); \
        v_store_(reinterpret_cast<void *>(angle1.data()), vf); \
        v_store_(reinterpret_cast<void *>(angle2.data()), vg); \
        v_store_(reinterpret_cast<void *>(angle3.data()), vh); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_MULEXPAND(_Tpvec, _Tpwvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(4*length); \
    vector<double> angle(length); \
    declare.in(X, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i+length); \
        _Tpwvec vc = v_load(reinterpret_cast<typename VTraits<_Tpwvec>::lane_type *>(X.data())+i+2*length); \
        _Tpwvec vd = v_load(reinterpret_cast<typename VTraits<_Tpwvec>::lane_type *>(X.data())+i+3*length); \
        startTimer(); \
        _Func(va, vb, vc, vd); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vc); \
        v_store_(reinterpret_cast<void *>(angle.data()), vd); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_ARG8I1O(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(4*length); \
    vector<double> angle(length); \
    declare.in(X, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i+length); \
        _Tpvec vc = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i+2*length); \
        _Tpvec vd = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i+3*length); \
        startTimer(); \
        auto vo = _Func(va, vb, vc, vd, va, vb, vc, vd); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vo); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_PACK_STORE(_Tp, _wTpvec, _Func) \
PERF_TEST_P(Intrinsic, _Func##_##_wTpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> angle(length); \
    declare.in(X, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_wTpvec>::vlanes()) { \
        _wTpvec va = v_load(reinterpret_cast<typename VTraits<_wTpvec>::lane_type *>(X.data())+i); \
        startTimer(); \
        _Func(reinterpret_cast<_Tp *>(angle.data()), va); \
        stopTimer(); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_RPACK_STORE_(_Tp, _wTpvec, _Func, _OFFSET) \
PERF_TEST_P(Intrinsic, _Func##_##_wTpvec##_OFFSET, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> angle(length); \
    declare.in(X, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_wTpvec>::vlanes()) { \
        _wTpvec va = v_load(reinterpret_cast<typename VTraits<_wTpvec>::lane_type *>(X.data())+i); \
        startTimer(); \
        _Func<_OFFSET>(reinterpret_cast<_Tp *>(angle.data()), va); \
        stopTimer(); \
    } \
    SANITY_CHECK_NOTHING(); \
}
#define OPENCV_HAL_PERF_TEST_RPACK_STORE(_OFFSET) \
OPENCV_HAL_PERF_TEST_RPACK_STORE_(uchar, v_uint16, v_rshr_pack_store, _OFFSET) \
OPENCV_HAL_PERF_TEST_RPACK_STORE_(schar, v_int16, v_rshr_pack_store, _OFFSET) \
OPENCV_HAL_PERF_TEST_RPACK_STORE_(ushort, v_uint32, v_rshr_pack_store, _OFFSET) \
OPENCV_HAL_PERF_TEST_RPACK_STORE_(short, v_int32, v_rshr_pack_store, _OFFSET) \
OPENCV_HAL_PERF_TEST_RPACK_STORE_(unsigned, v_uint64, v_rshr_pack_store, _OFFSET) \
OPENCV_HAL_PERF_TEST_RPACK_STORE_(int, v_int64, v_rshr_pack_store, _OFFSET) \
OPENCV_HAL_PERF_TEST_RPACK_STORE_(uchar, v_int16, v_rshr_pack_u_store, _OFFSET) \
OPENCV_HAL_PERF_TEST_RPACK_STORE_(ushort, v_int32, v_rshr_pack_u_store, _OFFSET)

#define OPENCV_HAL_PERF_TEST_rshr_pack_(_Tpvec, _Func, _OFFSET) \
PERF_TEST_P(Intrinsic, _Func##__##_Tpvec##_OFFSET, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(length); \
    vector<double> Y(length); \
    vector<double> angle(length); \
    declare.in(X, Y, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(Y.data())+i); \
        startTimer(); \
        auto vc = _Func<_OFFSET>(va, vb); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()), vc); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_rshr_pack(_OFFSET) \
OPENCV_HAL_PERF_TEST_rshr_pack_(v_uint16, v_rshr_pack, _OFFSET) \
OPENCV_HAL_PERF_TEST_rshr_pack_(v_int16, v_rshr_pack, _OFFSET) \
OPENCV_HAL_PERF_TEST_rshr_pack_(v_uint32, v_rshr_pack, _OFFSET) \
OPENCV_HAL_PERF_TEST_rshr_pack_(v_int32, v_rshr_pack, _OFFSET) \
OPENCV_HAL_PERF_TEST_rshr_pack_(v_uint64, v_rshr_pack, _OFFSET) \
OPENCV_HAL_PERF_TEST_rshr_pack_(v_int64, v_rshr_pack, _OFFSET) \
OPENCV_HAL_PERF_TEST_rshr_pack_(v_int16, v_rshr_pack_u, _OFFSET) \
OPENCV_HAL_PERF_TEST_rshr_pack_(v_int32, v_rshr_pack_u, _OFFSET)

#define OPENCV_HAL_PERF_TEST_INTERLEAVED(_Tpvec, _Func) \
PERF_TEST_P(Intrinsic, v_load_deinterleave##_2_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(2*length); \
    vector<double> angle(length); \
    vector<double> angle1(length); \
    declare.in(X, WARMUP_RNG).out(angle, angle1); \
    _Tpvec va, vb; \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        startTimer(); \
        v_load_deinterleave(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i*2, va, vb); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()+i), va); \
        v_store_(reinterpret_cast<void *>(angle1.data()+i), vb); \
    } \
    SANITY_CHECK_NOTHING(); \
} \
PERF_TEST_P(Intrinsic, v_load_deinterleave##_3_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(3*length); \
    vector<double> angle(length); \
    vector<double> angle1(length); \
    vector<double> angle2(length); \
    declare.in(X, WARMUP_RNG).out(angle, angle1, angle2); \
    _Tpvec va, vb, vc; \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        startTimer(); \
        v_load_deinterleave(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i*3, va, vb, vc); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()+i), va); \
        v_store_(reinterpret_cast<void *>(angle1.data()+i), vb); \
        v_store_(reinterpret_cast<void *>(angle2.data()+i), vc); \
    } \
    SANITY_CHECK_NOTHING(); \
} \
PERF_TEST_P(Intrinsic, v_load_deinterleave##_4_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X(4*length); \
    vector<double> angle(length); \
    vector<double> angle1(length); \
    vector<double> angle2(length); \
    vector<double> angle3(length); \
    declare.in(X, WARMUP_RNG).out(angle, angle1, angle2, angle3); \
    _Tpvec va, vb, vc, vd; \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        startTimer(); \
        v_load_deinterleave(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X.data())+i*4, va, vb, vc, vd); \
        stopTimer(); \
        v_store_(reinterpret_cast<void *>(angle.data()+i), va); \
        v_store_(reinterpret_cast<void *>(angle1.data()+i), vb); \
        v_store_(reinterpret_cast<void *>(angle2.data()+i), vc); \
        v_store_(reinterpret_cast<void *>(angle3.data()+i), vd); \
    } \
    SANITY_CHECK_NOTHING(); \
} \
PERF_TEST_P(Intrinsic, v_store_interleave##_2_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X1(length); \
    vector<double> X2(length); \
    vector<double> angle(2*length); \
    declare.in(X1, X2, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += 2*VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X1.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X2.data())+i); \
        startTimer(); \
        v_store_interleave(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(angle.data())+i*2, va, vb); \
        stopTimer(); \
    } \
    SANITY_CHECK_NOTHING(); \
} \
PERF_TEST_P(Intrinsic, v_store_interleave##_3_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X1(length); \
    vector<double> X2(length); \
    vector<double> X3(length); \
    vector<double> angle(3*length); \
    declare.in(X1,X2,X3, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X1.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X2.data())+i); \
        _Tpvec vc = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X3.data())+i); \
        startTimer(); \
        v_store_interleave(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(angle.data())+i*3, va, vb, vc); \
        stopTimer(); \
    } \
    SANITY_CHECK_NOTHING(); \
} \
PERF_TEST_P(Intrinsic, v_store_interleave##_4_##_Tpvec, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<double> X1(length); \
    vector<double> X2(length); \
    vector<double> X3(length); \
    vector<double> X4(length); \
    vector<double> angle(4*length); \
    declare.in(X1,X2,X3,X4, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i +=VTraits<_Tpvec>::vlanes()) { \
        _Tpvec va = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X1.data())+i); \
        _Tpvec vb = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X2.data())+i); \
        _Tpvec vc = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X3.data())+i); \
        _Tpvec vd = v_load(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(X4.data())+i); \
        startTimer(); \
        v_store_interleave(reinterpret_cast<typename VTraits<_Tpvec>::lane_type *>(angle.data())+i*4, va, vb, vc, vd); \
        stopTimer(); \
    } \
    SANITY_CHECK_NOTHING(); \
}

#define OPENCV_HAL_PERF_TEST_DISPATCH_ALL(_Macro, _Func) \
_Macro(v_uint8, _Func) \
_Macro(v_uint16, _Func) \
_Macro(v_uint32, _Func) \
_Macro(v_uint64, _Func) \
_Macro(v_int8, _Func) \
_Macro(v_int16, _Func) \
_Macro(v_int32, _Func) \
_Macro(v_int64, _Func) \
_Macro(v_float32, _Func) \
_Macro(v_float64, _Func)

#define OPENCV_HAL_PERF_TEST_DISPATCH_S(_Macro, _Func) \
_Macro(v_int8, _Func) \
_Macro(v_int16, _Func) \
_Macro(v_int32, _Func) \
_Macro(v_int64, _Func)

#define OPENCV_HAL_PERF_TEST_DISPATCH_U(_Macro, _Func) \
_Macro(v_uint8, _Func) \
_Macro(v_uint16, _Func) \
_Macro(v_uint32, _Func) \
_Macro(v_uint64, _Func)

#define OPENCV_HAL_PERF_TEST_DISPATCH_F(_Macro, _Func) \
_Macro(v_float32, _Func) \
_Macro(v_float64, _Func)

#define OPENCV_HAL_PERF_TEST_DISPATCH_L32(_Macro, _Func) \
_Macro(v_uint8, _Func) \
_Macro(v_uint16, _Func) \
_Macro(v_uint32, _Func) \
_Macro(v_int8, _Func) \
_Macro(v_int16, _Func) \
_Macro(v_int32, _Func) \
_Macro(v_float32, _Func)

#define OPENCV_HAL_PERF_TEST_DISPATCH_8(_Macro, _Func) \
_Macro(v_uint8, _Func) \
_Macro(v_int8, _Func)

#define OPENCV_HAL_PERF_TEST_DISPATCH_16(_Macro, _Func) \
_Macro(v_uint16, _Func) \
_Macro(v_int16, _Func)

#define OPENCV_HAL_PERF_TEST_DISPATCH_32(_Macro, _Func) \
_Macro(v_uint32, _Func) \
_Macro(v_int32, _Func) \
_Macro(v_float32, _Func)

#define OPENCV_HAL_PERF_TEST_DISPATCH_64(_Macro, _Func) \
_Macro(v_int64, _Func) \
_Macro(v_uint64, _Func) \
_Macro(v_float64, _Func)



// TODO: v_setzero_ v_setall_
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG1I1O, v_reinterpret_as_u8)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG1I1O, v_reinterpret_as_u16)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG1I1O, v_reinterpret_as_u32)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG1I1O, v_reinterpret_as_u64)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG1I1O, v_reinterpret_as_s8)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG1I1O, v_reinterpret_as_s16)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG1I1O, v_reinterpret_as_s32)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG1I1O, v_reinterpret_as_s64)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG1I1O, v_reinterpret_as_f32)

// TODO: v_extract v_extract_n v_extract_highest
// TODO: Load/Store

OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_LUT, v_lut)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_LUT, v_lut_pairs)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_LUT, v_lut_quads)

OPENCV_HAL_PERF_TEST_LUT_V(v_int32, v_lut)
// OPENCV_HAL_PERF_TEST_LUT_V(v_uint32, v_lut)
OPENCV_HAL_PERF_TEST_LUT_V(v_float32, v_lut)
OPENCV_HAL_PERF_TEST_LUT_V(v_float64, v_lut)

OPENCV_HAL_PERF_TEST_ARG2I1O(v_uint16, v_pack_b)
OPENCV_HAL_PERF_TEST_ARG4I1O(v_uint32, v_pack_b)
OPENCV_HAL_PERF_TEST_ARG8I1O(v_uint64, v_pack_b)

OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG2I1O, v_add)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG2I1O, v_sub)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I1O, v_mul)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_float64, v_mul)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_float64, v_div)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_float32, v_div)

OPENCV_HAL_PERF_TEST_MULEXPAND(v_uint8, v_uint16, v_mul_expand)
OPENCV_HAL_PERF_TEST_MULEXPAND(v_int8, v_int16, v_mul_expand)
OPENCV_HAL_PERF_TEST_MULEXPAND(v_uint16, v_uint32, v_mul_expand)
OPENCV_HAL_PERF_TEST_MULEXPAND(v_int16, v_int32, v_mul_expand)
OPENCV_HAL_PERF_TEST_MULEXPAND(v_uint32, v_uint64, v_mul_expand)

OPENCV_HAL_PERF_TEST_DISPATCH_16(OPENCV_HAL_PERF_TEST_ARG2I1O, v_mul_hi)

OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG2I1O, v_and)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG2I1O, v_or)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG2I1O, v_xor)
OPENCV_HAL_PERF_TEST_DISPATCH_S(OPENCV_HAL_PERF_TEST_ARG1I1O, v_not)
OPENCV_HAL_PERF_TEST_DISPATCH_U(OPENCV_HAL_PERF_TEST_ARG1I1O, v_not)

OPENCV_HAL_PERF_TEST_DISPATCH_16(OPENCV_HAL_PERF_TEST_SHIFT, v_shl)
OPENCV_HAL_PERF_TEST_DISPATCH_16(OPENCV_HAL_PERF_TEST_SHIFT, v_shr)
OPENCV_HAL_PERF_TEST_SHIFT(v_uint32, v_shl)
OPENCV_HAL_PERF_TEST_SHIFT(v_uint64, v_shl)
OPENCV_HAL_PERF_TEST_SHIFT(v_int32, v_shl)
OPENCV_HAL_PERF_TEST_SHIFT(v_int64, v_shl)
OPENCV_HAL_PERF_TEST_SHIFT(v_uint32, v_shr)
OPENCV_HAL_PERF_TEST_SHIFT(v_uint64, v_shr)
OPENCV_HAL_PERF_TEST_SHIFT(v_int32, v_shr)
OPENCV_HAL_PERF_TEST_SHIFT(v_int64, v_shr)

OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I1O, v_eq)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I1O, v_ne)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I1O, v_lt)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I1O, v_gt)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I1O, v_le)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I1O, v_ge)
OPENCV_HAL_PERF_TEST_DISPATCH_64(OPENCV_HAL_PERF_TEST_ARG2I1O, v_eq)
OPENCV_HAL_PERF_TEST_DISPATCH_64(OPENCV_HAL_PERF_TEST_ARG2I1O, v_ne)

OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I1O, v_min)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I1O, v_max)

OPENCV_HAL_PERF_TEST_DISPATCH_32(OPENCV_HAL_PERF_TEST_ARG8I0O, v_transpose4x4)

OPENCV_HAL_PERF_TEST_DISPATCH_S(OPENCV_HAL_PERF_TEST_REDUCE, v_reduce_sum)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_REDUCE, v_reduce_min)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_REDUCE, v_reduce_max)
// TODO: v_reduce_sum4

OPENCV_HAL_PERF_TEST_DISPATCH_F(OPENCV_HAL_PERF_TEST_ARG1I1O, v_sqrt)
OPENCV_HAL_PERF_TEST_DISPATCH_F(OPENCV_HAL_PERF_TEST_ARG1I1O, v_invsqrt)
OPENCV_HAL_PERF_TEST_DISPATCH_F(OPENCV_HAL_PERF_TEST_ARG2I1O, v_magnitude)
OPENCV_HAL_PERF_TEST_DISPATCH_F(OPENCV_HAL_PERF_TEST_ARG2I1O, v_sqr_magnitude)

OPENCV_HAL_PERF_TEST_DISPATCH_F(OPENCV_HAL_PERF_TEST_ARG3I0O, v_fma)
OPENCV_HAL_PERF_TEST_DISPATCH_F(OPENCV_HAL_PERF_TEST_ARG3I0O, v_muladd)
OPENCV_HAL_PERF_TEST_ARG3I0O(v_int32, v_fma)
OPENCV_HAL_PERF_TEST_ARG3I0O(v_int32, v_muladd)

OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_REDUCE, v_check_all)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_REDUCE, v_check_any)

OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I1O, v_absdiff)

OPENCV_HAL_PERF_TEST_ARG2I1O(v_int8, v_absdiffs)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int16,v_absdiffs)

OPENCV_HAL_PERF_TEST_DISPATCH_F(OPENCV_HAL_PERF_TEST_ARG1I1O, v_abs)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int8, v_abs)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int16, v_abs)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int32, v_abs)

OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_REDUCE_SAD, v_reduce_sad)

OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG3I1O, v_select)
OPENCV_HAL_PERF_TEST_ARG3I1O(v_float64, v_select)

OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ROTATE, v_rotate_right)
OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ROTATE, v_rotate_left)

OPENCV_HAL_PERF_TEST_ARG1I1O(v_int32, v_cvt_f32)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_float64, v_cvt_f32)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_float64, v_cvt_f32)

OPENCV_HAL_PERF_TEST_ARG1I1O(v_int32, v_cvt_f64)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_float32, v_cvt_f64)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int64, v_cvt_f64)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int32, v_cvt_f64_high)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_float32, v_cvt_f64_high)

OPENCV_HAL_PERF_TEST_ROTATE(v_uint32, v_broadcast_element)
OPENCV_HAL_PERF_TEST_ROTATE(v_int32, v_broadcast_element)
OPENCV_HAL_PERF_TEST_ROTATE(v_float32, v_broadcast_element)

OPENCV_HAL_PERF_TEST_ARG1I1O(v_uint32, v_broadcast_highest)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int32, v_broadcast_highest)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_float32, v_broadcast_highest)

OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_ARG1I1O, v_reverse)

OPENCV_HAL_PERF_TEST_ARG1I1O(v_uint8, v_expand_low)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int8, v_expand_low)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_uint16, v_expand_low)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int16, v_expand_low)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_uint32, v_expand_low)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int32, v_expand_low)

OPENCV_HAL_PERF_TEST_ARG1I1O(v_uint8, v_expand_high)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int8, v_expand_high)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_uint16, v_expand_high)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int16, v_expand_high)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_uint32, v_expand_high)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int32, v_expand_high)

OPENCV_HAL_PERF_TEST_LOAD(v_uint16, uchar, v_load_expand)
OPENCV_HAL_PERF_TEST_LOAD(v_int16, schar, v_load_expand)
OPENCV_HAL_PERF_TEST_LOAD(v_uint32, ushort, v_load_expand)
OPENCV_HAL_PERF_TEST_LOAD(v_int32, short, v_load_expand)
OPENCV_HAL_PERF_TEST_LOAD(v_uint64, uint, v_load_expand)
OPENCV_HAL_PERF_TEST_LOAD(v_int64, int, v_load_expand)
OPENCV_HAL_PERF_TEST_LOAD(v_uint32, uchar, v_load_expand_q)
OPENCV_HAL_PERF_TEST_LOAD(v_int32, schar, v_load_expand_q)

OPENCV_HAL_PERF_TEST_ARG2I1O(v_uint16, v_pack)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int16, v_pack)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_uint32, v_pack)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int32, v_pack)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_uint64, v_pack)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int64, v_pack)

OPENCV_HAL_PERF_TEST_PACK_STORE(uchar, v_uint16, v_pack_store)
OPENCV_HAL_PERF_TEST_PACK_STORE(schar, v_int16, v_pack_store)
OPENCV_HAL_PERF_TEST_PACK_STORE(ushort, v_uint32, v_pack_store)
OPENCV_HAL_PERF_TEST_PACK_STORE(short, v_int32, v_pack_store)
OPENCV_HAL_PERF_TEST_PACK_STORE(unsigned, v_uint64, v_pack_store)
OPENCV_HAL_PERF_TEST_PACK_STORE(int, v_int64, v_pack_store)

OPENCV_HAL_PERF_TEST_RPACK_STORE(1)
OPENCV_HAL_PERF_TEST_RPACK_STORE(2)
OPENCV_HAL_PERF_TEST_RPACK_STORE(8)


OPENCV_HAL_PERF_TEST_rshr_pack(1)
OPENCV_HAL_PERF_TEST_rshr_pack(2)
OPENCV_HAL_PERF_TEST_rshr_pack(8)

OPENCV_HAL_PERF_TEST_ARG2I1O(v_int32, v_pack_u)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int16, v_pack_u)

OPENCV_HAL_PERF_TEST_PACK_STORE(uchar, v_int16, v_pack_u_store)
OPENCV_HAL_PERF_TEST_PACK_STORE(ushort, v_int32, v_pack_u_store)

OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG4I0O, v_zip)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG4I0O, v_recombine)
OPENCV_HAL_PERF_TEST_ARG4I0O(v_float64, v_recombine)

OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I0O, v_combine_low)
OPENCV_HAL_PERF_TEST_ARG2I0O(v_float64, v_combine_low)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG2I0O, v_combine_high)
OPENCV_HAL_PERF_TEST_ARG2I0O(v_float64, v_combine_high)

OPENCV_HAL_PERF_TEST_DISPATCH_ALL(OPENCV_HAL_PERF_TEST_INTERLEAVED, foo)

OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG1I1O, v_interleave_pairs)
// OPENCV_HAL_PERF_TEST_ARG1I1O(v_uint32, v_interleave_quads)
// OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG1I1O, v_interleave_quads)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_uint8, v_interleave_quads)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int8, v_interleave_quads)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_uint16, v_interleave_quads)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_int16, v_interleave_quads)
OPENCV_HAL_PERF_TEST_DISPATCH_S(OPENCV_HAL_PERF_TEST_ARG1I1O, v_popcount)
OPENCV_HAL_PERF_TEST_DISPATCH_U(OPENCV_HAL_PERF_TEST_ARG1I1O, v_popcount)

// TODO: v_signmask v_scan_forward (rarely use)
OPENCV_HAL_PERF_TEST_DISPATCH_L32(OPENCV_HAL_PERF_TEST_ARG1I1O, v_pack_triplets)
// TODO: FP16 support

OPENCV_HAL_PERF_TEST_ARG1I1O(v_float32, v_round)
OPENCV_HAL_PERF_TEST_ARG1I1O(v_float64, v_round)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_float64, v_round)

OPENCV_HAL_PERF_TEST_DISPATCH_F(OPENCV_HAL_PERF_TEST_ARG1I1O, v_ceil)
OPENCV_HAL_PERF_TEST_DISPATCH_F(OPENCV_HAL_PERF_TEST_ARG1I1O, v_floor)
OPENCV_HAL_PERF_TEST_DISPATCH_F(OPENCV_HAL_PERF_TEST_ARG1I1O, v_trunc)

OPENCV_HAL_PERF_TEST_ARG2I1O(v_int16, v_dotprod)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int32, v_dotprod)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_uint8, v_dotprod_expand)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int8, v_dotprod_expand)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_uint16, v_dotprod_expand)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int16, v_dotprod_expand)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int32, v_dotprod_expand)

OPENCV_HAL_PERF_TEST_ARG2I1O(v_int16, v_dotprod_fast)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int32, v_dotprod_fast)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_uint8, v_dotprod_expand_fast)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int8, v_dotprod_expand_fast)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_uint16, v_dotprod_expand_fast)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int16, v_dotprod_expand_fast)
OPENCV_HAL_PERF_TEST_ARG2I1O(v_int32, v_dotprod_expand_fast)

PERF_TEST_P(Intrinsic, v_matmul_v_float32, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<float> X(length); \
    vector<float> angle(length); \
    declare.in(X, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<v_float32>::vlanes()) { \
        v_float32 va = v_load((X.data())+i); \
        startTimer(); \
        v_float32 res = v_matmul(va, va, va, va, va); \
        stopTimer(); \
        v_store_(angle.data(), res); \
    } \
    SANITY_CHECK_NOTHING(); \
}

PERF_TEST_P(Intrinsic, v_matmuladd_v_float32, testing::Values(CASES)) \
{ \
    size_t length = GetParam(); \
    vector<float> X(length); \
    vector<float> angle(length); \
    declare.in(X, WARMUP_RNG).out(angle); \
    for (size_t i = 0; i < length; i += VTraits<v_float32>::vlanes()) { \
        v_float32 va = v_load((X.data())+i); \
        startTimer(); \
        v_float32 res = v_matmuladd(va, va, va, va, va); \
        stopTimer(); \
        v_store_(angle.data(), res); \
    } \
    SANITY_CHECK_NOTHING(); \
}

} // namespace
