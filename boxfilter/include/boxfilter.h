// created by ftdlyc (yclu.cn@gmail.com)
// 2019.08.27

#pragma once
#ifndef BOXFILTER_H
#define BOXFILTER_H

#include <stdint.h>

extern "C" void boxfilter(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                          int width, int height, int stride_src, int stride_dst, int wsize);

#endif // BOXFILTER_H
