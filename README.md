# aqTFHE

Yet another reimplementation of TFHE in C++17.

About 12.5 ms/gate on Intel i7-8700.

## Caveat

`aqtfhe::native::secret_key::secret_key()` and other functions in aqTFHE
needs a random number generator as its argument.
**Use `std::random_device` there** if you don't care about it,
because it is supposed to be the only cryptographically secure
pseudo-random number generator (CSPRNG) in C++ standard library
(See [here](https://timsong-cpp.github.io/cppwp/n4659/rand) for the details).

## Licenses

This project is licensed under Apache License Version 2.0.
See the file LICENSE.

However the directory `spqlios/` is not my work but [TFHE](https://tfhe.github.io/tfhe/)'s one.
See the file `spqlios/LICENSE`.

## References

- [TFHE](https://tfhe.github.io/tfhe/)
- [TFHEpp](https://github.com/virtualsecureplatform/TFHEpp)
    - aqTFHE is strongly inspired by TFHEpp.
