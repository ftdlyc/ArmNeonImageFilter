// created by ftdlyc (yclu.cn@gmail.com)
// 2019.08.27

#pragma once
#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stdlib.h>

extern "C" void* aligned_malloc(size_t size);

extern "C" void* aligned_realloc(void* ptr, size_t size);

extern "C" void aligned_free(void* ptr);

#endif // ALLOCATOR_H
