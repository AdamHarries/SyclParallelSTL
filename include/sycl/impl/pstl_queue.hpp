
#ifndef __PSTL_QUEUE__
#define __PSTL_QUEUE__

#include <CL/sycl.hpp>


class pstl_queue
{
  cl::sycl::queue m_q;
public:
  pstl_queue(cl::sycl::queue a) = m_q(q){};
  ~pstl_queue();

  
};

#endif // __PSTL_QUEUE__ 

template <class inT, class outT>
class pstl_map 