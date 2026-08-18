#ifndef NUMPP_BACKEND_LAPACK_HPP
#define NUMPP_BACKEND_LAPACK_HPP

#if NUMPP_USE_LAPACK
#ifdef NUMPP_LAPACK_HEADR
    #include NUMPP_LAPACK_HEADER
#else
    #include <lapacke.h>
#endif

#endif

#endif