#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <string>
using namespace std::chrono;

#include "allocator.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_MALLOC(size) aligned_malloc(size)
#define STBI_REALLOC(ptr, size) aligned_realloc(ptr, size)
#define STBI_FREE(ptr) aligned_free(ptr)
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBW_MALLOC(size) aligned_malloc(size)
#define STBW_REALLOC(ptr, size) aligned_realloc(ptr, size)
#define STBW_FREE(ptr) aligned_free(ptr)
#include "stb_image_write.h"

#include "boxfilter.h"

int main(int argc, char** argv) {
  if(argc != 3) return 0;

  int width, height, channels;
  uint8_t* src = stbi_load(argv[1], &width, &height, &channels, 1);
  printf("input image: weight=%d, height=%d\n", width, height);

  uint8_t* dst = (uint8_t*)aligned_malloc(width * height);

  auto t1 = steady_clock::now();
  boxfilter(src, dst, width, height, width, width, std::stoi(argv[2]));
  auto t2 = steady_clock::now();
  printf("boxfilter took %lf ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.);

  stbi_write_png("./write.png", width, height, 1, dst, width);

  stbi_image_free(src);
  aligned_free(dst);

  return 0;
}
