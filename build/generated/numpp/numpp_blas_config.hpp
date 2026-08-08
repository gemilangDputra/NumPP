#pragma once

#define NUMPP_USE_BLAS 1

#if NUMPP_USE_BLAS
#define NUMPP_BLAS_HEADER <openblas/cblas.h>
#endif
