
#ifndef COMPLEX_KERNEL_H
#define COMPLEX_KERNEL_H
#include <math.h>
// A complex(ish) kernel for benchmarking the various foreach implementations
// float kernel(float val){
//   float acc = val;
//   // repeatedly do various bits of (hopefully) computationally 
//   // complicated maths
//   for(int i = 0;i<1000;i++){ 
//     if(i % 3 == 0){
//       acc = sin(acc * val);
//     }else{
//       if(i % 5 == 0){
//         acc = sqrt(acc + val/acc);
//       }
//     }
//   }
//   return acc;
// }

struct elem
{
  float values[128];
  float multiplicand;
  float initial;
};

// typedef typename float elem_t;
typedef struct elem elem_t;

elem_t make_element(int index){
  // return (float)index;
  elem_t s;
  for(int i = 0;i<128;i++){
    s.values[i] = i;
  }
  s.multiplicand = 2;
  s.initial = index;
  return s;
}

std::vector<elem_t> build_vector(int size){
  std::vector<elem_t> vect(size);
  for(int i = 0; i<size; i++){
    vect[i] = make_element(i);
  }
  return vect;
}

auto kernel = [](elem_t val){
  float acc = val.initial;
  for(int i = 0;i<128;i++){
    acc += (val.values[i])*val.multiplicand;
  }
  return acc;
};

// auto kernel_2 = [](elem_t val){
//   elem_t acc = val;
//   // repeatedly do various bits of (hopefully) computationally 
//   // complicated maths
//   for(int i = 0;i<1000;i++){ 
//     if(i % 3 == 0){
//       acc = cl::sycl::sin(acc * val + (val/acc) + (acc/val) / cl::sycl::atan(acc));
//     }else{
//       if(i % 5 == 0){
//         acc = cl::sycl::sqrt(acc + val/acc);
//       }
//     }
//   }
//   return acc;
// };

#endif