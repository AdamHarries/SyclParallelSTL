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
#ifndef __SYCL_IMPL_ABSTRACT_MAP__
#define __SYCL_IMPL_ABSTRACT_MAP__

#include <type_traits>
#include <algorithm>
#include <iostream>
#include <sycl/helpers/sycl_namegen.hpp>
#include <sycl/impl/pstl_queue.hpp>

namespace sycl {
namespace impl {

template <class ExecutionPolicy, class InT, class OutT, class UnaryOperation>
void sycl_transform_impl(ExecutionPolicy &sep, cl::sycl::buffer<InT, 1> &bufI, 
  cl::sycl::buffer<OutT, 1> &bufO,
  UnaryOperation op)
{
  cl::sycl::queue q(sep.get_queue());
  auto device = q.get_device();
  
  size_t length = bufI.get_count();
  auto local = std::min(device.get_info<cl::sycl::info::device::max_work_group_size>(),
               length);
  size_t global = sep.calculateGlobalSize(length, local);

  auto f = [length, local, global, &bufI, &bufO, op](
      cl::sycl::handler &h) mutable {
    cl::sycl::nd_range<3> r{
        cl::sycl::range<3>{std::max(global, local), 1, 1},
        cl::sycl::range<3>{local, 1, 1}};
    auto aI = bufI.template get_access<cl::sycl::access::mode::read>(h);
    auto aO = bufO.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<
        cl::sycl::helpers::NameGen<0, typename ExecutionPolicy::kernelName> >(
        r, [aI, aO, length, op](cl::sycl::nd_item<3> id) {
          int m_id = id.get_global(0);
          if(m_id < length){
            aO[m_id] = op(aI[m_id]);
          }
        });
  };
  q.submit(f);
}

template <class ExecutionPolicy, class InT, class OutT, class BinaryOperation>
void sycl_binary_transform_impl(ExecutionPolicy &sep, cl::sycl::buffer<InT, 1> &bufI1, 
  cl::sycl::buffer<InT, 1> &bufI2, cl::sycl::buffer<OutT, 1> &bufO, 
  BinaryOperation op)
{
  cl::sycl::queue q(sep.get_queue());
  auto device = q.get_device();

  size_t local = device.get_info<cl::sycl::info::device::max_work_group_size>();

  auto n = bufI1.get_count();

  size_t global = sep.calculateGlobalSize(n, local);

  auto f = [n, local, global, &bufI1, &bufI2, &bufO, op](cl::sycl::handler &h)
  mutable {
    cl::sycl::nd_range<3> r{cl::sycl::range<3>{std::max(global, local), 1, 1},
                            cl::sycl::range<3>{local, 1, 1}};
    auto a1 = bufI1.template get_access<cl::sycl::access::mode::read>(h);
    auto a2 = bufI2.template get_access<cl::sycl::access::mode::read>(h);
    auto aO = bufO.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<typename ExecutionPolicy::kernelName>(
        r, [a1, a2, aO, op, n](cl::sycl::nd_item<3> id) {
          if (id.get_global(0) < n) {
            aO[id.get_global(0)] =
                op(a1[id.get_global(0)], a2[id.get_global(0)]);
          }
        });
  };
  q.submit(f);
}

template <class ExecutionPolicy, class InT, class OutT, class BinaryOperation>
void sycl_zip_transform_impl(ExecutionPolicy &sep, cl::sycl::buffer<InT, 1> &bufI, 
  cl::sycl::buffer<OutT, 1> &bufO,
  BinaryOperation op)
{
  cl::sycl::queue q(sep.get_queue());
  auto device = q.get_device();
  
  size_t length = bufI.get_count();
  auto local = std::min(device.get_info<cl::sycl::info::device::max_work_group_size>(),
               length);
  size_t global = sep.calculateGlobalSize(length, local);

  auto f = [length, local, global, &bufI, &bufO, op](
      cl::sycl::handler &h) mutable {
    cl::sycl::nd_range<3> r{
        cl::sycl::range<3>{std::max(global, local), 1, 1},
        cl::sycl::range<3>{local, 1, 1}};
    auto aI = bufI.template get_access<cl::sycl::access::mode::read>(h);
    auto aO = bufO.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<
        cl::sycl::helpers::NameGen<0, typename ExecutionPolicy::kernelName> >(
        r, [aI, aO, length, op](cl::sycl::nd_item<3> id) {
          int m_id = id.get_global(0);
          if(m_id < length){
            aO[m_id] = op(aI[m_id], m_id);
          }
        });
  };
  q.submit(f);
}


}  // namespace impl
}  // namespace sycl


#endif  // __SYCL_IMPL_ABSTRACT_MAP__
