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
#ifndef __SYCL_IMPL_ABSTRACT_REDUCE__
#define __SYCL_IMPL_ABSTRACT_REDUCE__

#include <type_traits>
#include <algorithm>
#include <iostream>
#include <sycl/helpers/sycl_namegen.hpp>
#include <sycl/algorithm/algorithm_composite_patterns.hpp>

namespace sycl {
namespace impl {

/* In memory sycl reduction, given a specific buffer */

template <class ExecutionPolicy, class InT, class BinaryOperation>
void sycl_reduce_impl(ExecutionPolicy &sep, cl::sycl::buffer<InT, 1> &bufI, 
  BinaryOperation bop)
{
  cl::sycl::queue q(sep.get_queue());
  auto device = q.get_device();

  size_t length = bufI.get_count();
  auto local =
      std::min(device.get_info<cl::sycl::info::device::max_work_group_size>(),
               length);
  size_t global = sep.calculateGlobalSize(length, local);

  do {  
    auto f = [length, local, global, bufI, bop](cl::sycl::handler &h) mutable {
      cl::sycl::nd_range<3> r{cl::sycl::range<3>{std::max(global, local), 1,1},
                              cl::sycl::range<3>{local, 1,1}};
      auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
      cl::sycl::accessor<InT, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          scratch(cl::sycl::range<1>(local), h);

      h.parallel_for<
        cl::sycl::helpers::NameGen<1, typename ExecutionPolicy::kernelName, 
        class GenericReduceImpl > >(
        r, [aI, scratch, local, length, bop](cl::sycl::nd_item<3> id){
          int globalid = id.get_global(0);
          int localid = id.get_local(0);
          auto r = ReductionStrategy<InT>(local, length, id, scratch);
          r.workitem_get_from(aI);
          r.combine_threads(bop);
          r.workgroup_write_to(aI);          
        });
    };
    q.submit(f);
    length = length / local;
  } while (length > 1);
  q.wait_and_throw();
  // auto hI = bufI.template get_access<cl::sycl::access::mode::read,
  //                                    cl::sycl::access::target::host_buffer>();
}


}  // namespace impl
}  // namespace sycl


#endif  // __SYCL_IMPL_ABSTRACT_REDUCE__
