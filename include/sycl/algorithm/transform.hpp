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

#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_TRANSFORM__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_TRANSFORM__

#include <type_traits>
#include <algorithm>
#include <iostream>

// Detail header
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/impl/sycl_transform.hpp>

namespace sycl {
namespace impl {

/** transform sycl implementation
 * @brief Function that takes a Unary Operator and applies to the given range
 * @param sep : Execution Policy
 * @param b   : Start of the range
 * @param e   : End of the range
 * @param out : Output iterator
 * @param op  : Unary Operator
 * @return  An iterator pointing to the last element
 */
template <class ExecutionPolicy, class Iterator, class OutputIterator,
          class UnaryOperation>
OutputIterator transform(ExecutionPolicy &sep, Iterator b, Iterator e,
                         OutputIterator out, UnaryOperation op) {
  {
    auto bufI = sycl::helpers::make_const_buffer(b, e);
    auto bufO = sycl::helpers::make_buffer(out, out + bufI.get_count());

    sycl_transform_impl(sep, bufI, bufO, op);
  }
  return out;
}

/** transform sycl implementation
* @brief Function that takes a Binary Operator and applies to the given range
* @param sep    : Execution Policy
* @param first1 : Start of the range of buffer 1
* @param last1  : End of the range of buffer 1
* @param first2 : Start of the range of buffer 2
* @param result : Output iterator
* @param op     : Binary Operator
* @return  An iterator pointing to the last element
*/
template <class ExecutionPolicy, class InputIterator, class OutputIterator,
          class BinaryOperation>
OutputIterator transform(ExecutionPolicy &sep, InputIterator first1,
                         InputIterator last1, InputIterator first2,
                         OutputIterator result, BinaryOperation op) {
  auto buf1 = sycl::helpers::make_const_buffer(first1, last1);
  auto n = buf1.get_count();
  auto buf2 = sycl::helpers::make_const_buffer(first2, first2 + n);
  auto res = sycl::helpers::make_buffer(result, result + n);

  sycl_binary_transform_impl(sep, buf1, buf2, res, op);
  return first2 + n;
}

/** transform sycl implementation
* @brief Function that takes a Binary Operator and applies to the given range
* @param sep    : Execution Policy
* @param q      : Queue
* @param first1 : Start of the range of buffer 1
* @param last1  : End of the range of buffer 1
* @param first2 : Start of the range of buffer 2
* @param result : Output iterator
* @param op     : Binary Operator
* @return  An iterator pointing to the last element
*/
template <class ExecutionPolicy, class InputIterator, class OutputIterator,
          class BinaryOperation>
OutputIterator transform(ExecutionPolicy &sep, cl::sycl::queue &q,
                         InputIterator first1, InputIterator last1,
                         InputIterator first2, OutputIterator result,
                         BinaryOperation op) {
  auto buf1 = sycl::helpers::make_const_buffer(first1, last1);
  auto n = buf1.get_count();
  auto buf2 = sycl::helpers::make_const_buffer(first2, first2 + n);
  auto res = sycl::helpers::make_buffer(result, result + n);

  sycl_binary_transform_impl(sep, buf1, buf2, res, op);
  return first2 + n;
}

/** transform sycl implementation
* @brief Function that takes a Binary Operator and applies to the given range
* @param sep    : Execution Policy
* @param q      : Queue
* @param buf1   : buffer 1
* @param buf2   : buffer 2
* @param res    : Output buffer
* @param op     : Binary Operator
* @return  An iterator pointing to the last element
*/
template <class ExecutionPolicy, class Buffer, class BinaryOperation>
void transform(ExecutionPolicy &sep, cl::sycl::queue &q, Buffer &buf1,
               Buffer &buf2, Buffer &res, BinaryOperation op) {
  sycl_binary_transform_impl(sep, buf1, buf2, res, op);
}

}  // namespace impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_TRANSFORM__
