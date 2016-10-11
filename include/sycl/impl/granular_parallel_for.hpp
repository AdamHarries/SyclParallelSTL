/*
 * Copyright (c) 2015 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
*/

/**
 * @file
 * @brief 
 */

#ifndef _EXPERIMENTAL_DETAIL_SYCL_GRANULAR_PARALLEL_FOR__
#define _EXPERIMENTAL_DETAIL_SYCL_GRANULAR_PARALLEL_FOR__

namespace cl {
namespace sycl {
namespace helpers {

/*
  A parallel for implementation with a variable task granularity per thread
*/

template <typename nameT = std::nullptr_t, typename functorT>
void parallel_for(cl::sycl::handler &h,
                  const cl::sycl::nd_range<3> pRange,
                  const size_t lRange,
                  functorT functor) {
  // run the functor with the thread structure given by pRange, but appearing to
  // the functor as if lRange threads (in one continuous run of threads) are being run 
  
  // build a wrapping functor that translates between physical and logical ranges
  auto f = [functor, lRange](cl::sycl::nd_item<3> pId) {
    // Get the physical (global) thread id, and global range
    auto global = pId.get_global(0);
    auto global_range = pId.get_global_range(0);

    // iterate a logical id through the logical range, starting with the physical
    // id, and incrementing it by the physical range while still within the 
    // logical range
    for(auto id = global; id < lRange; id += global_range){
      functor(id);
    }
  };

  // call the parallel for as usual, with the physical range, and the new functor
  h.parallel_for<nameT>(pRange,f);
}

/*
  A more granular parallel for, using the handler to suggest a good local/global
  size
*/
template <typename nameT = std::nullptr_t, typename functorT, typename ExecutionPolicy>
void parallel_for(cl::sycl::handler &h, 
                  ExecutionPolicy &sep,
                  const size_t lRange,
                  functorT functor){
  // we handle this all within the parallel_for, as if we do it outside, we might
  // missuse the information, and call the normal parallel_for with it. 
  auto pRange = sep.calculateGranularDimensions(lRange);
  cl::sycl::helpers::parallel_for<nameT>(h, pRange, lRange, functor);
}

}
}
}

#endif  // _EXPERIMENTAL_DETAIL_SYCL_GRANULAR_PARALLEL_FOR__
