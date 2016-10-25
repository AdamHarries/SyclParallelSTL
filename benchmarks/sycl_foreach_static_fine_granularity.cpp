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

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

#include <experimental/algorithm>
#include <sycl/execution_policy>

#include "benchmark.h"
#include "complex_kernel.h"

using namespace sycl::helpers;

template <class ExecutionPolicy, class Iterator, class UnaryFunction>
void for_each_impl(ExecutionPolicy &sep, Iterator b, Iterator e,
                   UnaryFunction op) {
  {
    cl::sycl::queue q(sep.get_queue());
    auto device = q.get_device();
    size_t localRange =
        device.get_info<cl::sycl::info::device::max_work_group_size>();
    typedef typename std::iterator_traits<Iterator>::value_type type_;
    auto bufI = sycl::helpers::make_buffer(b, e);
    auto vectorSize = bufI.get_count();
    size_t globalRange = sep.calculateGlobalSize(vectorSize, localRange);
    auto f = [vectorSize, localRange, globalRange, &bufI, op, &sep](
        cl::sycl::handler &h) mutable {
      // range is same as input size - fine granularity
      cl::sycl::nd_range<3> r{
          cl::sycl::range<3>{std::max(globalRange, localRange), 1, 1},
          cl::sycl::range<3>{localRange, 1, 1}};
      auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
      // chunking parallel for using sizes given by implementation
      cl::sycl::helpers::parallel_for<typename ExecutionPolicy::kernelName>(
          h, r, vectorSize, [aI, op, vectorSize](size_t id) { op(aI[id]); });

    };
    q.submit(f);
  }
}

benchmark<>::time_units_t benchmark_foreach_static_fine(
    const unsigned numReps, const unsigned num_elems,
    const cli_device_selector cds) {
  auto v1 = build_vector(num_elems);

  cl::sycl::queue q(cds);
  auto myforeach = [&]() {
    sycl::sycl_execution_policy<class ForEachAlgorithm1> snp(q);
    for_each_impl(snp, begin(v1), end(v1), kernel);
  };

  auto time = benchmark<>::duration(numReps, myforeach);

  return time;
}

BENCHMARK_MAIN("BENCH_SYCL_FOREACH_STATIC_FINE", benchmark_foreach_static_fine,
               2u, 33554432u, 10);
