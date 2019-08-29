// created by ftdlyc (yclu.cn@gmail.com)
// 2019.08.27

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__i386__) || defined(__x86_64__)
#include "NEON_2_SSE.h"

inline uint8x16_t vshrn_high_n_u16(uint8x8_t v0l, uint16x8_t v1, const int n) {
  uint8x8_t v0h = vshrn_n_u16(v1, n);
  uint8x16_t v0 = vcombine_u8(v0l, v0h);
  return v0;
}

inline uint16x8_t vshrn_high_n_u32(uint16x4_t v0l, uint32x4_t v1, const int n) {
  uint16x4_t v0h = vshrn_n_u32(v1, n);
  uint16x8_t v0 = vcombine_u16(v0l, v0h);
  return v0;
}
#else
#include <arm_neon.h>
#endif

#include "allocator.h"
#include "boxfilter.h"

template<uint16_t b>
inline uint16x8_t vdivq_n_u16(uint16x8_t a) {
  if constexpr(b == 1) return a;
  if constexpr(b == 2) return vshrq_n_u16(a, 1);
  constexpr uint16_t n = (uint16_t)ceil(log2(b));
  constexpr uint16_t m = 1 + ((((uint32_t)1 << n) - b) << 16) / b;
  constexpr uint16_t shift1 = 1;
  constexpr uint16_t shift2 = n - shift1;

  uint16x4_t vm = vdup_n_u16(m);

  // t1 = (a * m) >> 16
  // 16bit * 16bit -> 32bit, extract high 16bit
  uint16x4_t v0l = vget_low_u16(a);
  uint16x4_t v0h = vget_high_u16(a);
  uint32x4_t v1l = vmull_u16(v0l, vm);
  uint32x4_t v1h = vmull_u16(v0h, vm);
  uint16x4_t t1l = vshrn_n_u32(v1l, 16);
  uint16x8_t t1 = vshrn_high_n_u32(t1l, v1h, 16);

  // t2 = a - t1
  // t3 = t2 >> shift1
  // t4 = t1 + t3
  // t5 = t4 >> shift2
  uint16x8_t t2 = vsubq_u16(a, t1);
  uint16x8_t t3 = vshrq_n_u16(t2, shift1);
  uint16x8_t t4 = vaddq_u16(t1, t3);
  uint16x8_t t5 = vshrq_n_u16(t4, shift2);

  return t5;
}

template<uint16_t b>
inline uint16x4_t vdiv_n_u16(uint16x4_t a) {
  if constexpr(b == 1) return a;
  if constexpr(b == 2) return vshr_n_u16(a, 1);
  constexpr uint16_t n = (uint16_t)ceil(log2(b));
  constexpr uint16_t m = 1 + ((((uint32_t)1 << n) - b) << 16) / b;
  constexpr uint16_t shift1 = 1;
  constexpr uint16_t shift2 = n - shift1;

  uint16x4_t vm = vdup_n_u16(m);

  // t1 = (a * m) >> 16
  // 16bit * 16bit -> 32bit, extract high 16bit
  uint32x4_t v0 = vmull_u16(a, vm);
  uint16x4_t t1 = vshrn_n_u32(v0, 16);

  // t2 = a - t1
  // t3 = t2 >> shift1
  // t4 = t1 + t3
  // t5 = t4 >> shift2
  uint16x4_t t2 = vsub_u16(a, t1);
  uint16x4_t t3 = vshr_n_u16(t2, shift1);
  uint16x4_t t4 = vadd_u16(t1, t3);
  uint16x4_t t5 = vshr_n_u16(t4, shift2);

  return t5;
}

inline void transpose8x8_u16(const uint16x8_t& v0, const uint16x8_t& v1, const uint16x8_t& v2, const uint16x8_t& v3,
                             const uint16x8_t& v4, const uint16x8_t& v5, const uint16x8_t& v6, const uint16x8_t& v7,
                             uint16x8_t& v8, uint16x8_t& v9, uint16x8_t& v10, uint16x8_t& v11,
                             uint16x8_t& v12, uint16x8_t& v13, uint16x8_t& v14, uint16x8_t& v15) {
  uint16x8_t v16 = vuzp1q_u16(v0, v1); // 00 02 04 06 10 12 14 16
  uint16x8_t v17 = vuzp2q_u16(v0, v1); // 01 03 05 07 11 13 15 17
  uint16x8_t v18 = vuzp1q_u16(v2, v3); // 20 22 24 26 30 32 34 36
  uint16x8_t v19 = vuzp2q_u16(v2, v3); // 21 23 25 27 31 33 35 37
  uint16x8_t v20 = vuzp1q_u16(v4, v5); // 40 42 44 46 50 52 54 56
  uint16x8_t v21 = vuzp2q_u16(v4, v5); // 41 43 45 47 51 53 55 57
  uint16x8_t v22 = vuzp1q_u16(v6, v7); // 60 62 64 66 70 72 74 76
  uint16x8_t v23 = vuzp2q_u16(v6, v7); // 61 63 65 67 71 73 75 77

  uint16x8_t v24 = vuzp1q_u16(v16, v18); // 00 04 10 14 20 24 30 34
  uint16x8_t v25 = vuzp2q_u16(v16, v18); // 02 06 12 16 22 26 32 36
  uint16x8_t v26 = vuzp1q_u16(v20, v22); // 40 44 50 54 60 64 70 74
  uint16x8_t v27 = vuzp2q_u16(v20, v22); // 42 46 52 56 62 66 72 76

  v8 = vuzp1q_u16(v24, v26);  // 00 10 20 30 40 50 60 70
  v10 = vuzp1q_u16(v25, v27); // 02 12 22 32 42 52 62 72
  v12 = vuzp2q_u16(v24, v26); // 04 14 24 34 44 54 64 74
  v14 = vuzp2q_u16(v25, v27); // 06 16 26 36 46 56 66 76

  v24 = vuzp1q_u16(v17, v19); // 01 05 11 15 21 25 31 35
  v25 = vuzp2q_u16(v17, v19); // 03 07 13 17 23 27 33 37
  v26 = vuzp1q_u16(v21, v23); // 41 45 51 55 61 65 71 75
  v27 = vuzp2q_u16(v21, v23); // 43 47 53 57 63 67 73 77

  v9 = vuzp1q_u16(v24, v26);  // 01 11 21 31 41 51 61 71
  v11 = vuzp1q_u16(v25, v27); // 03 13 23 33 43 53 63 73
  v13 = vuzp2q_u16(v24, v26); // 05 15 25 35 45 55 65 75
  v15 = vuzp2q_u16(v25, v27); // 07 17 27 37 47 57 67 77
}

inline void transpose4x4_u16(const uint16x4_t& v0, const uint16x4_t& v1, const uint16x4_t& v2, const uint16x4_t& v3,
                             uint16x4_t& v4, uint16x4_t& v5, uint16x4_t& v6, uint16x4_t& v7) {
  uint16x4_t v8 = vuzp1_u16(v0, v1);  // 00 02 10 12
  uint16x4_t v9 = vuzp2_u16(v0, v1);  // 01 03 11 13
  uint16x4_t v10 = vuzp1_u16(v2, v3); // 20 22 30 32
  uint16x4_t v11 = vuzp2_u16(v2, v3); // 21 23 31 33

  v4 = vuzp1_u16(v8, v10); // 00 10 20 30
  v5 = vuzp1_u16(v9, v11); // 01 11 21 31
  v6 = vuzp2_u16(v8, v10); // 02 12 22 32
  v7 = vuzp2_u16(v9, v11); // 03 13 23 33
}

template<int radius>
void boxfilter_u16_neon(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                        int width, int height, int stride_src, int stride_dst) {
  uint16_t* buf0 = (uint16_t*)aligned_malloc(width * sizeof(uint16_t));
  uint16_t* buf1 = (uint16_t*)aligned_malloc(width * sizeof(uint16_t));
  constexpr uint16_t divisor = (2 * radius + 1) * (2 * radius + 1);

  // first row
  {
    const uint8_t* src__ = src;
    uint16_t* buf0__ = buf0;
    for(int i = 0; i < width / 16; ++i) {
      uint8x16_t v0 = vld1q_u8(src__);
      uint16x8_t v1 = vmovl_u8(vget_low_u8(v0));
      uint16x8_t v2 = vmovl_u8(vget_high_u8(v0));
      vst1q_u16(buf0__, v1);
      vst1q_u16(buf0__ + 8, v2);

      src__ += 16;
      buf0__ += 16;
    }

    int remain = width & 0xF;
    if(remain >= 8) {
      uint8x8_t v0 = vld1_u8(src__);
      uint16x8_t v1 = vshll_n_u8(v0, 0);
      vst1q_u16(buf0__, v1);

      src__ += 8;
      buf0__ += 8;
      remain -= 8;
    }
    if(remain > 0) {
      for(; remain > 0; --remain) {
        *buf0__ = *src__;

        ++src__;
        ++buf0__;
      }
    }

    for(int j = 1; j < 2 * radius + 1; ++j) {
      const uint8_t* src__ = src + j * stride_src;
      uint16_t* buf0__ = buf0;
      for(int i = 0; i < width / 16; ++i) {
        uint8x16_t v0 = vld1q_u8(src__);
        uint16x8_t v1 = vld1q_u16(buf0__);
        uint16x8_t v2 = vld1q_u16(buf0__ + 8);
        uint16x8_t v3 = vmovl_u8(vget_low_u8(v0));
        uint16x8_t v4 = vmovl_u8(vget_high_u8(v0));
        v1 = vaddq_u16(v1, v3);
        v2 = vaddq_u16(v2, v4);
        vst1q_u16(buf0__, v1);
        vst1q_u16(buf0__ + 8, v2);

        src__ += 16;
        buf0__ += 16;
      }

      int remain = width & 0xF;
      if(remain >= 8) {
        uint8x8_t v0 = vld1_u8(src__);
        uint16x8_t v1 = vld1q_u16(buf0__);
        uint16x8_t v2 = vshll_n_u8(v0, 0);
        v1 = vaddq_u16(v1, v2);
        vst1q_u16(buf0__, v1);

        src__ += 8;
        buf0__ += 8;
        remain -= 8;
      }
      if(remain > 0) {
        for(; remain > 0; --remain) {
          *buf0__ += *src__;

          ++src__;
          ++buf0__;
        }
      }
    }
  }

  for(int j = radius; j < height - 1 - radius; ++j) {
    const uint8_t* src0__ = src + (j - radius) * stride_src;
    const uint8_t* src1__ = src + (j + radius + 1) * stride_src;
    uint8_t* dst__ = dst + j * stride_dst;
    uint16_t* buf0__ = buf0;
    uint16_t* buf1__ = buf1;

    buf1[radius] = 0;
    for(int i = 0; i < 2 * radius + 1; ++i) {
      buf1[radius] += buf0__[i];
    }
    for(int i = radius + 1; i < width - radius; ++i) {
      buf1[i] = buf1[i - 1] + buf0__[i + radius] - buf0__[i - radius - 1];
    }

    // cal mean
    for(int i = 0; i < width / 16; ++i) {
      uint16x8_t v0l = vld1q_u16(buf1__);
      uint16x8_t v0h = vld1q_u16(buf1__ + 8);
      uint16x8_t v1l = vdivq_n_u16<divisor>(v0l);
      uint16x8_t v1h = vdivq_n_u16<divisor>(v0h);
      uint8x8_t v2l = vmovn_u16(v1l);
      uint8x8_t v2h = vmovn_u16(v1h);

      vst1_u8(dst__, v2l);
      vst1_u8(dst__ + 8, v2h);

      buf1__ += 16;
      dst__ += 16;
    }

    int remain = width & 0xF;
    if(remain >= 8) {
      uint16x8_t v0 = vld1q_u16(buf1__);
      uint16x8_t v1 = vdivq_n_u16<divisor>(v0);
      uint8x8_t v2 = vmovn_u16(v1);

      vst1_u8(dst__, v2);

      buf1__ += 8;
      dst__ += 8;
      remain -= 8;
    }
    if(remain > 0) {
      for(; remain > 0; --remain) {
        *dst__ = *buf1__ / divisor;

        ++buf1__;
        ++dst__;
      }
    }

    // int remain = width & 0xF;
    for(int i = 0; i < width / 16; ++i) {
      uint8x16_t v0 = vld1q_u8(src0__);
      uint8x16_t v1 = vld1q_u8(src1__);
      uint16x8_t v2 = vld1q_u16(buf0__);
      uint16x8_t v3 = vld1q_u16(buf0__ + 8);

      uint16x8_t v4 = vmovl_u8(vget_low_u8(v0));
      uint16x8_t v5 = vmovl_u8(vget_high_u8(v0));
      uint16x8_t v6 = vmovl_u8(vget_low_u8(v1));
      uint16x8_t v7 = vmovl_u8(vget_high_u8(v1));

      v2 = vsubq_u16(v2, v4);
      v3 = vsubq_u16(v3, v5);
      v2 = vaddq_u16(v2, v6);
      v3 = vaddq_u16(v3, v7);

      vst1q_u16(buf0__, v2);
      vst1q_u16(buf0__ + 8, v3);

      src0__ += 16;
      src1__ += 16;
      buf0__ += 16;
    }

    remain = width & 0xF;
    if(remain >= 8) {
      uint8x8_t v0 = vld1_u8(src0__);
      uint8x8_t v1 = vld1_u8(src1__);
      uint16x8_t v2 = vld1q_u16(buf0__);

      uint16x8_t v3 = vmovl_u8(v0);
      uint16x8_t v4 = vmovl_u8(v1);

      v2 = vsubq_u16(v2, v3);
      v2 = vaddq_u16(v2, v4);

      vst1q_u16(buf0__, v2);

      src0__ += 8;
      src1__ += 8;
      buf0__ += 8;
      remain -= 8;
    }
    if(remain > 0) {
      for(; remain > 0; --remain) {
        *buf0__ = *buf0__ - *src1__ + *src0__;

        ++src0__;
        ++src1__;
        ++buf0__;
      }
    }
  }

  // last row
  {
    uint8_t* dst__ = dst + (height - 1 - radius) * stride_dst;
    uint16_t* buf0__ = buf0;
    uint16_t* buf1__ = buf1;

    buf1[radius] = 0;
    for(int i = 0; i < 2 * radius + 1; ++i) {
      buf1[radius] += buf0__[i];
    }
    for(int i = radius + 1; i < width - radius; ++i) {
      buf1[i] = buf1[i - 1] + buf0__[i + radius] - buf0__[i - radius - 1];
    }

    // cal mean
    for(int i = 0; i < width / 16; ++i) {
      uint16x8_t v0l = vld1q_u16(buf1__);
      uint16x8_t v0h = vld1q_u16(buf1__ + 8);
      uint16x8_t v1l = vdivq_n_u16<divisor>(v0l);
      uint16x8_t v1h = vdivq_n_u16<divisor>(v0h);
      uint8x8_t v2l = vmovn_u16(v1l);
      uint8x8_t v2h = vmovn_u16(v1h);

      vst1_u8(dst__, v2l);
      vst1_u8(dst__ + 8, v2h);

      buf1__ += 16;
      dst__ += 16;
    }

    int remain = width & 0xF;
    if(remain >= 8) {
      uint16x8_t v0 = vld1q_u16(buf1__);
      uint16x8_t v1 = vdivq_n_u16<divisor>(v0);
      uint8x8_t v2 = vmovn_u16(v1);

      vst1_u8(dst__, v2);

      buf1__ += 8;
      dst__ += 8;
      remain -= 8;
    }
    if(remain > 0) {
      for(; remain > 0; --remain) {
        *dst__ = *buf1__ / divisor;

        ++buf1__;
        ++dst__;
      }
    }
  }

  aligned_free(buf0);
  aligned_free(buf1);
}

void transpose_image_u16(const uint16_t* __restrict__ src, uint16_t* __restrict__ dst,
                         int width, int height, int stride_src, int stride_dst) {
}

void boxfilter(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
               int width, int height, int stride_src, int stride_dst, int radius) {
  switch(radius) {
  case 1:
    boxfilter_u16_neon<1>(src, dst, width, height, stride_src, stride_dst);
    break;
  case 2:
    boxfilter_u16_neon<2>(src, dst, width, height, stride_src, stride_dst);
    break;
  case 3:
    boxfilter_u16_neon<3>(src, dst, width, height, stride_src, stride_dst);
    break;
  case 4:
    boxfilter_u16_neon<4>(src, dst, width, height, stride_src, stride_dst);
    break;
  case 5:
    boxfilter_u16_neon<5>(src, dst, width, height, stride_src, stride_dst);
    break;
  case 6:
    boxfilter_u16_neon<6>(src, dst, width, height, stride_src, stride_dst);
    break;
  case 7:
    boxfilter_u16_neon<7>(src, dst, width, height, stride_src, stride_dst);
    break;
  default:
    printf("boxfilter error: only support radius <=7!\n");
    break;
  }
}
