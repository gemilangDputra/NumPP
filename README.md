# NumPP
### Numeric++ — Modern C++ Numerical Array Library
> **v1.0.0 · Unstable · Active Development**
![C](https://img.shields.io/badge/C-A8B9CC?style=flat-square&logo=c&logoColor=white)![C++](https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=c%2B%2B&logoColor=white)![CMake](https://img.shields.io/badge/CMake-064F8C?style=flat-square&logo=cmake&logoColor=white)

**NumPP** (*Numeric++*) is a modern C++ numerical computing library focused
on multidimensional data, generic abstractions, and flexible numerical operations.

NumPP is designed to work beyond its own data structures. Through genericinterfaces and views, user-defined types can be integrated into NumPP's
numerical operations without requiring them to directly inherit from or become a NumPP container.

The API is designed around modern C++ concepts such as generic programming, 
type safety, views, and extensible computational backends.

> ⚠️ **Project Status**
>
> NumPP is still in the early stages of development. The API and implementation are subject to change, and patches are actively being applied.

## Features
- Multidimensional array storage
- Generic dan type-safe API
- Modern C++ design
- Pluggable computational kernels
- BLAS backend support
- CMake-based build system

## Performance Kernel
Native NumPP is currently still under development and **performance is not yet optimal**.

For high-performance use cases, it is recommended to use **BLAS kernels** in accordance with the configuration guidelines.

> **TL;DR:** Native kernel → development / testing  
> BLAS kernel → recommended for performance

## Requirements
- C++20 or later
- CMake 3.20+
- BLAS (optional and recommended)

## Installation
Coming soon.
## Usage
Coming soon.

## Roadmap
- [ ] Improve native kernel performance
- [x] Expand multidimensional operations
- [x] Improve BLAS integration
- [ ]  Expand test coverage
- [ ]  Benchmarking suite
- [ ]  Documentation

## Contributing

Contributions, bug reports, benchmarks, and feedback are welcome.

## License
See `LICENSE` for details.
