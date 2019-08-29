// created by ftdlyc (yclu.cn@gmail.com)
// 2019.08.27

#include <malloc.h>
#include <stdint.h>
#include <stdlib.h>

#include "allocator.h"

#ifndef ALIGN_BYTES
#define ALIGN_BYTES 16
#endif

inline size_t align_hi(size_t size, size_t align) {
  return (size + align - 1) & ~(align - 1);
}

inline void* align_ptr(void* ptr, size_t align) {
  return (void*)(((size_t)ptr + align - 1) & ~(align - 1));
}

void* aligned_malloc(size_t size) {
  if(size <= 0) return NULL;

  uint8_t* ptr = (uint8_t*)malloc(size + ALIGN_BYTES + sizeof(uint8_t*));
  if(ptr == NULL) return ptr;

  uint8_t** aptr = (uint8_t**)align_ptr(ptr + sizeof(uint8_t*), ALIGN_BYTES);
  aptr[-1] = ptr;

  return aptr;
}

void* aligned_realloc(void* ptr, size_t size) {
  if(ptr) {
    uint8_t* ptr_old = ((uint8_t**)ptr)[-1];
    size_t shift_old = (size_t)ptr - (size_t)ptr_old;

    uint8_t* ptr_new = (uint8_t*)realloc(ptr_old, size + ALIGN_BYTES + sizeof(uint8_t*));
    if(ptr_old == ptr_new) return ptr_new;

    uint8_t** aptr = (uint8_t**)align_ptr(ptr_new + sizeof(uint8_t*), ALIGN_BYTES);
    size_t shift_new = (size_t)aptr - (size_t)ptr_new;
    aptr[-1] = ptr_new;

    size_t old_size = malloc_usable_size(ptr_old) - ALIGN_BYTES - sizeof(uint8_t*);
    ptr_old = ptr_new + shift_old;
    ptr_new = ptr_new + shift_new;
    if(shift_old < shift_new) {
      for(int i = old_size - 1; i >= 0; --i) ptr_new[i] = ptr_old[i];
    } else if(shift_old > shift_new) {
      for(int i = 0; i < old_size; ++i) ptr_new[i] = ptr_old[i];
    }

    return aptr;
  }
  return NULL;
}

void aligned_free(void* ptr) {
  if(ptr) {
    free(((void**)ptr)[-1]);
  }
}
