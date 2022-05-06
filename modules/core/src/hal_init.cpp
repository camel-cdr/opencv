#if 0
#include "precomp.hpp"
#include "opencv2/core/hal/intrin_rvv_vec.hpp"

namespace cv
{
namespace hal_baseline
{
namespace rvv
{
    inline unsigned int VTraits<v_uint8>::nlanes = vsetvlmax_e8m1();
    inline unsigned int VTraits<v_int8>::nlanes = vsetvlmax_e8m1();
    inline unsigned int VTraits<v_uint16>::nlanes = vsetvlmax_e16m1();
    inline unsigned int VTraits<v_int16>::nlanes = vsetvlmax_e16m1();
    inline unsigned int VTraits<v_float32>::nlanes = vsetvlmax_e32m1();
    inline unsigned int VTraits<v_uint32>::nlanes = vsetvlmax_e32m1();
    inline unsigned int VTraits<v_int32>::nlanes = vsetvlmax_e32m1();
    inline unsigned int VTraits<v_uint64>::nlanes = vsetvlmax_e64m1();
    inline unsigned int VTraits<v_int64>::nlanes = vsetvlmax_e64m1();
    #if CV_SIMD128_64F
    inline unsigned int VTraits<v_float64>::nlanes = vsetvlmax_e64m1();
    #endif
}
}  
}
#endif
