# BRANCH DESCRIPTION, DELETE ON MERGE

Internal ticket: task/9065

Description: A number of different algorithms, such as find_if, minmax, reduce etc all use a similar map-reduce style implementation with varying operators to compute their results. At present, each algorithm essentially re-implements the operations in the reduce, as we cannot call parallel STL functions with sycl iterators.

The core of the reduction/find_if/minmax algorithms should be refactored into (say) the algorithm_composite_patterns module to enable code reuse.


