
#ifndef __PSTL_QUEUE__
#define __PSTL_QUEUE__

#include <CL/sycl.hpp>


// class pstl_queue
// {
//   cl::sycl::queue m_q;
// public:
//   pstl_queue(cl::sycl::queue a) = m_q(q){};
//   ~pstl_queue();

  
// };



template <typename InT, typename OutT> class Map;
template <typename InT, typename OutT> class Reduce;


template <typename ...Values> class Seq;

template <typename Map, typename Reduce> 
class Seq<Map, Reduce> {
  void eval(Map m, Reduce r){
    printf("Map then Reduce\n");
  }
};

template <typename Map1, typename Map2> 
class Seq<Map1, Map2> {
  void eval(Map1 m1, Map2 m2){
    printf("Map1 then Map2\n");
  }
};

#endif // __PSTL_QUEUE__ 