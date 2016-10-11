/* Copyright (c) 2015 The Khronos Group Inc.

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

#ifndef __SYCL_IMPL_ALGORITHM_TRANSFORM_REDUCE__
#define __SYCL_IMPL_ALGORITHM_TRANSFORM_REDUCE__

#include <type_traits>
#include <algorithm>
#include <iostream>

// SYCL helpers header
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_differences.hpp>
#include <sycl/algorithm/algorithm_composite_patterns.hpp>
#include <sycl/impl/sycl_reduce.hpp>
#include <sycl/impl/sycl_transform.hpp>

namespace sycl {
namespace impl {

/* transform_reduce.
* @brief Returns the transform_reduce of one vector across the range [first1,
* last1) by applying Functions op1 and op2. Implementation of the command
* group
* that submits a transform_reduce kernel.
*/
template <class ExecutionPolicy, class InputIterator, class UnaryOperation,
          class T, class BinaryOperation>
T transform_reduce(ExecutionPolicy& exec, InputIterator first,
                   InputIterator last, UnaryOperation unary_op, T init,
                   BinaryOperation binary_op) {

  auto vectorSize = sycl::helpers::distance(first, last);


  // build buffer to hold initial data
  auto bufI = sycl::helpers::make_const_buffer(first, last);

  // build buffer to hold result of transform
  cl::sycl::buffer<T, 1> bufR((cl::sycl::range<1>(vectorSize)));
  if (vectorSize < 1) {
    return init;
  }

  // transform from bufI -> bufR using unary op 
  sycl_transform_impl(exec, bufI, bufR, unary_op);

  // reduce bufR using binary_op
  sycl_reduce_impl(exec, bufR, binary_op);

  // get accessor to the buffer
  auto hR = bufR.template get_access<cl::sycl::access::mode::read,
                                     cl::sycl::access::target::host_buffer>();

  return binary_op(hR[0], init);
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_TRANSFORM_REDUCE__
