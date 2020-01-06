#ifndef ANQOU_AQTFHE_HPP
#define ANQOU_AQTFHE_HPP

#include <array>
#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

//
#include <spqlios-fft.h>

namespace aqtfhe {

namespace native {

template <class T, size_t Dim>
class vector_slice {
private:
    std::array<size_t, Dim> shape_;
    typename std::vector<T>::iterator head_;
    size_t size_;

public:
    vector_slice(std::array<size_t, Dim> shape,
                 typename std::vector<T>::iterator head)
        : shape_(std::move(shape)),
          head_(head),
          size_(std::accumulate(shape.begin(), shape.end(), 1,
                                std::multiplies<size_t>{}))
    {
    }

    vector_slice(std::vector<T>& data)
        : shape_({data.size()}), head_(data.begin()), size_(data.size())
    {
        static_assert(Dim == 1);
    }

    std::array<size_t, Dim> shape() const { return shape_; }

    size_t size() const { return size_; }

    T* data() { return &*head_; }
    const T* data() const { return &*head_; }

    decltype(auto) operator[](size_t index)
    {
        if constexpr (Dim == 1) {
            return *(head_ + index);
        }
        else {
            // offset = index * shape_[1] * shape_[2] * ... * shape_[Dim-1]
            size_t offset = size_ / shape_[0] * index;

            // new_shape[0...Dim-1] = shape[1...Dim]
            std::array<size_t, Dim - 1> new_shape;
            for (size_t i = 1; i < Dim; i++)
                new_shape[i - 1] = shape_[i];

            return vector_slice<T, Dim - 1>(new_shape, head_ + offset);
        }
    }

    vector_slice<T, Dim>& operator=(const std::vector<T>& rhs)
    {
        assert(size_ == rhs.size());
        std::copy(rhs.begin(), rhs.end(), head_);
        return *this;
    }
};

template <class T, size_t Dim>
class const_vector_slice {
private:
    std::array<size_t, Dim> shape_;
    typename std::vector<T>::const_iterator head_;
    size_t size_;

public:
    const_vector_slice(std::array<size_t, Dim> shape,
                       typename std::vector<T>::const_iterator head)
        : shape_(std::move(shape)),
          head_(head),
          size_(std::accumulate(shape.begin(), shape.end(), 1,
                                std::multiplies<size_t>{}))
    {
    }

    const_vector_slice(const std::vector<T>& data)
        : shape_({data.size()}), head_(data.begin()), size_(data.size())
    {
        static_assert(Dim == 1);
    }

    std::array<size_t, Dim> shape() const { return shape_; }

    size_t size() const { return size_; }

    const T* data() const { return &*head_; }

    decltype(auto) operator[](size_t index) const
    {
        if constexpr (Dim == 1) {
            return *(head_ + index);
        }
        else {
            // offset = index * shape_[1] * shape_[2] * ... * shape_[Dim-1]
            size_t offset = size_ / shape_[0] * index;

            // new_shape[0...Dim-1] = shape[1...Dim]
            std::array<size_t, Dim - 1> new_shape;
            for (size_t i = 1; i < Dim; i++)
                new_shape[i - 1] = shape_[i];

            return const_vector_slice<T, Dim - 1>(new_shape, head_ + offset);
        }
    }
};

template <class T, size_t Dim>
class nested_vector {
private:
    std::array<size_t, Dim> shape_;
    std::vector<T> data_;

public:
    template <class... Shape>
    nested_vector(Shape... shape)
        : shape_({static_cast<size_t>(shape)...}),
          data_((static_cast<size_t>(shape) * ...))
    {
        static_assert(Dim != 0);
        static_assert(sizeof...(Shape) == Dim);
    }

    std::array<size_t, Dim> shape() const { return shape_; }

    std::vector<T> get() const { return data_; }

    decltype(auto) operator[](size_t index)
    {
        return vector_slice<T, Dim>(shape_, data_.begin())[index];
    }

    decltype(auto) operator[](size_t index) const
    {
        return const_vector_slice<T, Dim>(shape_, data_.begin())[index];
    }
};

template <class T, class... Shape>
auto make_nested_vector(Shape... shape)
{
    return nested_vector<T, sizeof...(Shape)>(shape...);
}

using uint32vec = std::vector<uint32_t>;
using uint64vec = std::vector<uint64_t>;
using doublevec = std::vector<double>;
using doublevec_view = const_vector_slice<double, 1>;
using uint32vec_view = const_vector_slice<uint32_t, 1>;

struct lwe_params {
    uint32_t n;
    double alpha;
    uint32_t Nbit;
    uint32_t N;
    uint32_t l;
    uint32_t Bgbit;
    uint32_t Bg;
    double alphabk;
    uint32_t t;
    uint32_t basebit;
    double alphaks;
    uint32_t mu;

    uint32_t nbarbit;
    uint32_t nbar;
    uint32_t lbar;
    uint32_t Bgbitbar;
    uint32_t Bgbar;
    double alphabklvl02;
    uint32_t tbar;
    uint32_t basebitlvl21;
    double alphaprivks;
    uint64_t mubar;

    static lwe_params tfhe_default()
    {
        const uint32_t n = 500;
        const double alpha = 2.44e-5;
        const uint32_t Nbit = 10;
        const uint32_t N = 1 << Nbit;
        const uint32_t l = 2;
        const uint32_t Bgbit = 10;
        const uint32_t Bg = 1 << Bgbit;
        const double alphabk = 3.73e-9;
        const uint32_t t = 8;
        const uint32_t basebit = 2;
        const double alphaks = 2.44e-5;
        const uint32_t mu = 1U << 29;

        const uint32_t nbarbit = 11;
        const uint32_t nbar = 1 << nbarbit;
        const uint32_t lbar = 4;
        const uint32_t Bgbitbar = 9;
        const uint32_t Bgbar = 1 << Bgbitbar;
        const double alphabklvl02 = std::pow(2.0, -44);
        const uint32_t tbar = 10;
        const uint32_t basebitlvl21 = 3;
        const double alphaprivks = std::pow(2, -31);
        const uint64_t mubar = 1UL << 61;

        return lwe_params{
            n,       alpha,        Nbit,        N,        l,       Bgbit,
            Bg,      alphabk,      t,           basebit,  alphaks, mu,
            nbarbit, nbar,         lbar,        Bgbitbar, Bgbar,   alphabklvl02,
            tbar,    basebitlvl21, alphaprivks, mubar,
        };
    }
};

class cloud_key;
class secret_key;

namespace detail {
inline uint32_t dtot32(double d)
{
    return int32_t(int64_t((d - int64_t(d)) * (1L << 32)));
}

inline doublevec mul_in_fd(uint32_t N, doublevec_view a, doublevec_view b)
{
    doublevec ret(N);

    for (uint32_t i = 0; i < N / 2; i++) {
        double aimbim = a[i + N / 2] * b[i + N / 2];
        double arebim = a[i] * b[i + N / 2];
        ret[i] = a[i] * b[i] - aimbim;
        ret[i + N / 2] = a[i + N / 2] * b[i] + arebim;
    }

    return ret;
}

inline void fma_in_fd(uint32_t N, doublevec& res, doublevec_view a,
                      doublevec_view b)
{
    for (uint32_t i = 0; i < N / 2; i++) {
        res[i] = a[i + N / 2] * b[i + N / 2] - res[i];
        res[i] = a[i] * b[i] - res[i];
        res[i + N / 2] += a[i] * b[i + N / 2];
        res[i + N / 2] += a[i + N / 2] * b[i];
    }
}

class fft_processor {
private:
    uint32_t N_;
    FFT_Processor_Spqlios spqlios_;

public:
    fft_processor(uint32_t N) : N_(N), spqlios_(N) {}

    template <class T>
    uint32vec twist_fft_lvl1(T&& a)
    {
        assert(a.size() == N_);  // FIXME: relax this condition.
        uint32vec ret(N_);
        spqlios_.execute_direct_torus32(ret.data(), a.data());
        return ret;
    }

    template <class T>
    doublevec twist_ifft_lvl1(T&& a)
    {
        assert(a.size() == N_);  // FIXME: relax this condition.
        doublevec ret(N_);
        spqlios_.execute_reverse_torus32(ret.data(), a.data());
        return ret;
    }
};

class tlwe_lvl0;
class tlwe_lvl1;
class trlwe_lvl1;

class key_switching_key {
private:
    nested_vector<uint32_t, 4> data_;  // N * t * (1 << basebit - 1) * (n + 1)

public:
    template <class RandomEngine>
    key_switching_key(const lwe_params& params, RandomEngine& randeng,
                      const uint32vec& keylvl0, const uint32vec& keylvl1);

    // FIXME: delete
    key_switching_key(const nested_vector<uint32_t, 4>& data) : data_(data) {}

    const_vector_slice<uint32_t, 3> operator[](size_t i) const
    {
        return data_[i];
    }
};

class trgswfft_lvl1 {
private:
    const lwe_params& params_;
    nested_vector<double, 3> data_;

public:
    trgswfft_lvl1(const lwe_params& params, nested_vector<double, 3> data)
        : params_(params), data_(std::move(data))
    {
        assert(2 * params_.l == data_.shape()[0]);
        assert(2 == data_.shape()[1]);
        assert(params_.N == data_.shape()[2]);
    }

    template <class RandomEngine>
    static trgswfft_lvl1 sym_encrypt(const lwe_params& params,
                                     const uint32vec& keylvl1,
                                     RandomEngine& randeng, int32_t p);

    const_vector_slice<double, 2> operator[](uint32_t i) const
    {
        return data_[i];
    }
};

class trlwe_in_fd_lvl1 {
private:
    const lwe_params& params_;
    doublevec poly0_, poly1_;

public:
    trlwe_in_fd_lvl1(const lwe_params& params, doublevec poly0, doublevec poly1)
        : params_(params), poly0_(std::move(poly0)), poly1_(std::move(poly1))
    {
        assert(params_.N == poly0_.size());
        assert(params_.N == poly1_.size());
    }

    trlwe_lvl1 twist_fft() const;
};

class decomposed_trlwe_in_fd_lvl1 {
private:
    const lwe_params& params_;
    nested_vector<double, 2> decvecfft_;

public:
    decomposed_trlwe_in_fd_lvl1(const lwe_params& params,
                                nested_vector<double, 2> decvecfft)
        : params_(params), decvecfft_(std::move(decvecfft))
    {
        assert(decvecfft_.shape()[0] == 2 * params_.l);
        assert(decvecfft_.shape()[1] == params_.N);
    }

    trlwe_in_fd_lvl1 mul_and_fma_in_fd(const trgswfft_lvl1& trgswfft) const;
};

class trlwe_lvl1 {
private:
    const lwe_params& params_;
    uint32vec poly0_, poly1_;

public:
    trlwe_lvl1(const lwe_params& params, uint32vec poly0, uint32vec poly1)
        : params_(params), poly0_(std::move(poly0)), poly1_(std::move(poly1))
    {
        assert(params_.N == poly0_.size());
        assert(params_.N == poly1_.size());
    }

    uint32vec& poly0() { return poly0_; }
    uint32vec& poly1() { return poly1_; }
    const uint32vec& poly0() const { return poly0_; }
    const uint32vec& poly1() const { return poly1_; }

    trlwe_lvl1 polynomial_mul_by_xai_minus_one(uint32_t a) const;
    trlwe_lvl1 trgswfft_external_product(const trgswfft_lvl1& trgswfft) const;
    tlwe_lvl1 sample_extract_index(uint32_t index) const;
    decomposed_trlwe_in_fd_lvl1 decomposition_and_ifft() const;

    template <class RandomEngine>
    static trlwe_lvl1 sym_encrypt_zero(const lwe_params& params,
                                       const uint32vec& keylvl1,
                                       RandomEngine& randeng);
    template <class RandomEngine>
    static trlwe_lvl1 sym_encrypt(const lwe_params& params,
                                  const uint32vec& keylvl1,
                                  RandomEngine& randeng, const uint32vec& p);
    uint32vec sym_decrypt(const uint32vec& keylvl1) const;

private:
    static uint32vec poly_mul_lvl1(const uint32vec& a, const uint32vec& b)
    {
        assert(a.size() == b.size());
        const uint32_t N = a.size();
        thread_local detail::fft_processor fftproc(N);

        doublevec ffta = fftproc.twist_ifft_lvl1(a);
        doublevec fftb = fftproc.twist_ifft_lvl1(b);
        return fftproc.twist_fft_lvl1(mul_in_fd(N, ffta, fftb));
    }

    static uint32vec polynomial_mul_by_xai_minus_one(uint32_t N, uint32_t a,
                                                     const uint32vec& poly);
};

class tlwe_lvl1 {
private:
    const lwe_params& params_;
    uint32vec tlwe_;

public:
    tlwe_lvl1(const lwe_params& params, uint32vec tlwe)
        : params_(params), tlwe_(std::move(tlwe))
    {
        assert(tlwe_.size() == params_.N + 1);
    }

    tlwe_lvl0 identity_key_switch(const key_switching_key& ksk) const;
};

class tlwe_lvl0 {
private:
    const lwe_params& params_;
    uint32vec tlwe_;

public:
    tlwe_lvl0(const lwe_params& params, uint32vec tlwe)
        : params_(params), tlwe_(std::move(tlwe))
    {
        assert(tlwe_.size() == params_.n + 1);
    }

    const uint32vec& get() const { return tlwe_; }

    template <class RandomEngine>
    static tlwe_lvl0 boots_sym_encrypt(const lwe_params& params,
                                       RandomEngine& randeng,
                                       const uint32vec& keylvl0, bool val);
    bool boots_sym_decrypt(const uint32vec& keylvl0) const;

    template <class RandomEngine>
    static tlwe_lvl0 sym_encrypt(const lwe_params& params,
                                 const uint32vec& keylvl0,
                                 RandomEngine& randeng, uint32_t p);
    bool sym_decrypt(const uint32vec& keylvl0) const;

    tlwe_lvl0 gate_bootstrapping(const std::vector<trgswfft_lvl1>& bkfftlvl01,
                                 const key_switching_key& ksk) const;

    tlwe_lvl0 nand(const tlwe_lvl0& rhs,
                   const std::vector<trgswfft_lvl1>& bkfftlvl01,
                   const key_switching_key& ksk) const;

private:
    tlwe_lvl1 gate_bootstrapping_to_lvl1(
        const std::vector<trgswfft_lvl1>& bkfftlvl01) const;
};

}  // namespace detail

class secret_key;

class cloud_key {
private:
    lwe_params params_;
    detail::key_switching_key ksk_;
    std::vector<detail::trgswfft_lvl1> bkfftlvl01_;

public:
    template <class RandomEngine>
    cloud_key(const lwe_params& params, RandomEngine& randeng,
              const uint32vec& keylvl0, const uint32vec& keylvl1)
        : params_(params), ksk_(params, randeng, keylvl0, keylvl1)
    {
        // Fill bkfftlvl01_
        bkfftlvl01_.reserve(params_.n);
        for (size_t i = 0; i < params_.n; i++) {
            bkfftlvl01_.push_back(detail::trgswfft_lvl1::sym_encrypt(
                params_, keylvl1, randeng, static_cast<uint32_t>(keylvl0[i])));
        }
    }

    const detail::key_switching_key& ksk() const { return ksk_; }
    const std::vector<detail::trgswfft_lvl1>& bkfftlvl01() const
    {
        return bkfftlvl01_;
    }
};

class secret_key {
private:
    lwe_params params_;
    uint32vec keylvl0_;  // n
    uint32vec keylvl1_;  // N

public:
    template <class RandomEngine>
    secret_key(lwe_params params, RandomEngine& randeng)
        : params_(std::move(params)), keylvl0_(params.n), keylvl1_(params.N)
    {
        std::bernoulli_distribution binary(0.5);
        for (uint32_t& val : keylvl0_)
            val = binary(randeng);
        for (uint32_t& val : keylvl1_)
            val = binary(randeng);
    }

    const lwe_params& params() const { return params_; }
    const uint32vec& keylvl0() const { return keylvl0_; }
    const uint32vec& keylvl1() const { return keylvl1_; }

    template <class RandomEngine>
    cloud_key get_cloud_key(RandomEngine& randeng) const
    {
        return cloud_key{params_, randeng, keylvl0_, keylvl1_};
    }
};

class encrypted_bit {
private:
    detail::tlwe_lvl0 tlwe_;

private:
    encrypted_bit(detail::tlwe_lvl0 tlwe) : tlwe_(std::move(tlwe)) {}

public:
    template <class RandomEngine>
    encrypted_bit(const secret_key& skey, RandomEngine& randeng, bool val)
        : tlwe_(detail::tlwe_lvl0::boots_sym_encrypt(skey.params(), randeng,
                                                     skey.keylvl0(), val))
    {
    }

    bool decrypt(const secret_key& skey) const
    {
        return tlwe_.boots_sym_decrypt(skey.keylvl0());
    }

    encrypted_bit nand(const encrypted_bit& rhs, const cloud_key& ck)
    {
        return encrypted_bit{tlwe_.nand(rhs.tlwe_, ck.bkfftlvl01(), ck.ksk())};
    }
};

namespace detail {
template <class RandomEngine>
tlwe_lvl0 tlwe_lvl0::boots_sym_encrypt(const lwe_params& params,
                                       RandomEngine& randeng,
                                       const uint32vec& keylvl0, bool val)
{
    return tlwe_lvl0::sym_encrypt(params, keylvl0, randeng,
                                  val ? params.mu : -params.mu);
}

template <class RandomEngine>
tlwe_lvl0 tlwe_lvl0::sym_encrypt(const lwe_params& params,
                                 const uint32vec& keylvl0,
                                 RandomEngine& randeng, uint32_t p)
{
    std::uniform_int_distribution<uint32_t> torus(
        0, std::numeric_limits<uint32_t>::max());
    std::normal_distribution<double> gaussian(0., params.alpha);

    uint32vec tlwe(params.n + 1);
    tlwe[params.n] = p + dtot32(gaussian(randeng));
    for (uint32_t i = 0; i < params.n; i++) {
        tlwe[i] = torus(randeng);
        tlwe[params.n] += tlwe[i] * keylvl0[i];
    }

    return tlwe_lvl0{params, tlwe};
}

bool tlwe_lvl0::boots_sym_decrypt(const uint32vec& keylvl0) const
{
    return sym_decrypt(keylvl0);
}

bool tlwe_lvl0::sym_decrypt(const uint32vec& keylvl0) const
{
    uint32_t phase = tlwe_[params_.n];
    for (uint32_t i = 0; i < params_.n; i++)
        phase -= tlwe_[i] * keylvl0[i];
    return static_cast<int32_t>(phase) > 0;
}

template <class RandomEngine>
trlwe_lvl1 trlwe_lvl1::sym_encrypt_zero(const lwe_params& params,
                                        const uint32vec& keylvl1,
                                        RandomEngine& randeng)
{
    assert(keylvl1.size() == params.N);

    std::uniform_int_distribution<uint32_t> torus(
        0, std::numeric_limits<uint32_t>::max());
    std::normal_distribution<double> gaussian(0., params.alphabk);

    uint32vec poly0(params.N);
    for (uint32_t& val : poly0)
        val = torus(randeng);
    uint32vec poly1 = poly_mul_lvl1(poly0, keylvl1);
    for (uint32_t& val : poly1) {
        double err = gaussian(randeng);
        val += dtot32(err);
    }

    return trlwe_lvl1{params, poly0, poly1};
}

template <class RandomEngine>
trlwe_lvl1 trlwe_lvl1::sym_encrypt(const lwe_params& params,
                                   const uint32vec& keylvl1,
                                   RandomEngine& randeng, const uint32vec& p)
{
    assert(p.size() == params.N);

    trlwe_lvl1 ret = sym_encrypt_zero(params, keylvl1, randeng);
    for (uint32_t i = 0; i < params.N; i++)
        ret.poly1_[i] += p[i];

    return ret;
}

uint32vec trlwe_lvl1::sym_decrypt(const uint32vec& keylvl1) const
{
    uint32vec mulres = poly_mul_lvl1(poly0_, keylvl1);
    uint32vec phase = poly1_;
    for (uint32_t i = 0; i < params_.N; i++)
        phase[i] -= mulres[i];

    uint32vec p(params_.N);
    for (uint32_t i = 0; i < params_.N; i++)
        p[i] = static_cast<int32_t>(phase[i]) > 0;

    return p;
}

trlwe_lvl1 trlwe_lvl1::trgswfft_external_product(
    const trgswfft_lvl1& trgswfft) const
{
    decomposed_trlwe_in_fd_lvl1 decvecfft = decomposition_and_ifft();
    trlwe_in_fd_lvl1 trlwe = decvecfft.mul_and_fma_in_fd(trgswfft);
    return trlwe.twist_fft();
}

trlwe_in_fd_lvl1 decomposed_trlwe_in_fd_lvl1::mul_and_fma_in_fd(
    const trgswfft_lvl1& trgswfft) const
{
    doublevec poly0 = mul_in_fd(params_.N, decvecfft_[0], trgswfft[0][0]);
    doublevec poly1 = mul_in_fd(params_.N, decvecfft_[0], trgswfft[0][1]);
    for (uint32_t i = 1; i < 2 * params_.l; i++) {
        fma_in_fd(params_.N, poly0, decvecfft_[i], trgswfft[i][0]);
        fma_in_fd(params_.N, poly1, decvecfft_[i], trgswfft[i][1]);
    }
    return trlwe_in_fd_lvl1{params_, std::move(poly0), std::move(poly1)};
}

decomposed_trlwe_in_fd_lvl1 trlwe_lvl1::decomposition_and_ifft() const
{
    const uint32_t Bgbit = params_.Bgbit, N = params_.N, l = params_.l,
                   Bg = params_.Bg;

    // offsetgen
    uint32_t offset = 0;
    for (uint32_t i = 1; i <= l; i++)
        offset += Bg / 2 * (1U << (32 - i * Bgbit));

    // decomposition
    uint32_t mask = static_cast<uint32_t>((1UL << Bgbit) - 1);
    auto decvec = make_nested_vector<uint32_t>(2 * l, N);
    for (uint32_t i = 0; i < N; i++) {
        decvec[0][i] = poly0_[i] + offset;
        decvec[l][i] = poly1_[i] + offset;
    }

    uint32_t halfBg = (1UL << (Bgbit - 1));
    for (int i = l - 1; i >= 0; i--) {
        for (uint32_t j = 0; j < N; j++) {
            decvec[i][j] =
                ((decvec[0][j] >> (32 - (i + 1) * Bgbit)) & mask) - halfBg;
            decvec[i + l][j] =
                ((decvec[l][j] >> (32 - (i + 1) * Bgbit)) & mask) - halfBg;
        }
    }

    // twist ifft
    thread_local detail::fft_processor fftproc(N);
    auto decvecfft = make_nested_vector<double>(2 * l, N);
    for (uint32_t i = 0; i < 2 * l; i++)
        decvecfft[i] = fftproc.twist_ifft_lvl1(decvec[i]);

    return decomposed_trlwe_in_fd_lvl1{params_, decvecfft};
}

template <class RandomEngine>
trgswfft_lvl1 trgswfft_lvl1::sym_encrypt(const lwe_params& params,
                                         const uint32vec& keylvl1,
                                         RandomEngine& randeng, int32_t p)
{
    // trgsw sym encrypt
    // FIXME: more efficient implementation?
    std::vector<trlwe_lvl1> trgsw;
    trgsw.reserve(2 * params.l);
    for (size_t i = 0; i < 2 * params.l; i++)
        trgsw.push_back(trlwe_lvl1::sym_encrypt_zero(params, keylvl1, randeng));
    for (uint32_t i = 0; i < params.l; i++) {
        uint32_t h = 1U << (32 - (i + 1) * params.Bgbit);
        trgsw[i].poly0()[0] += static_cast<uint32_t>(p) * h;
        trgsw[i + params.l].poly1()[0] += static_cast<uint32_t>(p) * h;
    }

    // trgswfft sym encrypt
    thread_local detail::fft_processor fftproc(params.N);
    auto trgswfft = make_nested_vector<double>(2 * params.l, 2, params.N);
    for (uint32_t i = 0; i < 2 * params.l; i++) {
        trgswfft[i][0] = fftproc.twist_ifft_lvl1(trgsw[i].poly0());
        trgswfft[i][1] = fftproc.twist_ifft_lvl1(trgsw[i].poly1());
    }

    return trgswfft_lvl1{params, std::move(trgswfft)};
}

trlwe_lvl1 trlwe_in_fd_lvl1::twist_fft() const
{
    thread_local detail::fft_processor fftproc(params_.N);
    return trlwe_lvl1{params_, fftproc.twist_fft_lvl1(poly0_),
                      fftproc.twist_fft_lvl1(poly1_)};
}

tlwe_lvl0 tlwe_lvl1::identity_key_switch(const key_switching_key& ksk) const
{
    const uint32_t n = params_.n, N = params_.N, basebit = params_.basebit,
                   t = params_.t;
    const uint32_t prec_offset = 1U << (32 - (1 + basebit * t));
    const uint32_t mask = (1U << basebit) - 1;

    uint32vec ret(n + 1, 0);
    ret[n] = tlwe_[N];
    for (uint32_t i = 0; i < N; i++) {
        const uint32_t aibar = tlwe_[i] + prec_offset;
        for (uint32_t j = 0; j < t; j++) {
            const uint32_t aij = (aibar >> (32 - (j + 1) * basebit)) & mask;
            if (aij == 0)
                continue;
            for (uint32_t k = 0; k <= n; k++)
                ret[k] -= ksk[i][j][aij - 1][k];
        }
    }

    return tlwe_lvl0{params_, std::move(ret)};
}

inline uint32_t mod_switch_from_torus32(uint32_t Msize, uint32_t phase)
{
    uint64_t interv = ((1UL << 63) / Msize) * 2;  // width of each intervall
    uint64_t half_interval = interv / 2;  // begin of the first intervall
    uint64_t phase64 = (uint64_t(phase) << 32) + half_interval;
    // floor to the nearest multiples of interv
    return static_cast<uint32_t>(phase64 / interv);
}

inline trlwe_lvl1 rotated_test_vector(const lwe_params& params, uint32_t bara)
{
    const uint32_t N = params.N, mu = params.mu;
    uint32vec poly1(N);

    if (bara < N) {
        for (uint32_t i = 0; i < bara; i++)
            poly1[i] = -mu;
        for (uint32_t i = bara; i < N; i++)
            poly1[i] = mu;
    }
    else {
        const uint32_t baraa = bara - N;
        for (uint32_t i = 0; i < baraa; i++)
            poly1[i] = mu;
        for (uint32_t i = baraa; i < N; i++)
            poly1[i] = -mu;
    }

    return trlwe_lvl1{params, uint32vec(N, 0), poly1};
}

uint32vec trlwe_lvl1::polynomial_mul_by_xai_minus_one(uint32_t N, uint32_t a,
                                                      const uint32vec& poly)
{
    if (a == 0)
        return poly;

    uint32vec res(N);
    if (a < N) {
        for (uint32_t i = 0; i < a; i++)
            res[i] = -poly[i - a + N] - poly[i];
        for (uint32_t i = a; i < N; i++)
            res[i] = poly[i - a] - poly[i];
    }
    else {
        const uint32_t aa = a - N;
        for (uint32_t i = 0; i < aa; i++)
            res[i] = poly[i - aa + N] - poly[i];
        for (uint32_t i = aa; i < N; i++)
            res[i] = -poly[i - aa] - poly[i];
    }

    return res;
}

trlwe_lvl1 trlwe_lvl1::polynomial_mul_by_xai_minus_one(uint32_t a) const
{
    return trlwe_lvl1{params_,
                      polynomial_mul_by_xai_minus_one(params_.N, a, poly0_),
                      polynomial_mul_by_xai_minus_one(params_.N, a, poly1_)};
}

tlwe_lvl0 tlwe_lvl0::gate_bootstrapping(
    const std::vector<trgswfft_lvl1>& bkfftlvl01,
    const key_switching_key& ksk) const
{
    return gate_bootstrapping_to_lvl1(bkfftlvl01).identity_key_switch(ksk);
}

tlwe_lvl1 tlwe_lvl0::gate_bootstrapping_to_lvl1(
    const std::vector<trgswfft_lvl1>& bkfftlvl01) const
{
    const uint32_t n = params_.n, N = params_.N;

    uint32_t bara = 2 * N - mod_switch_from_torus32(2 * N, tlwe_[n]);
    trlwe_lvl1 acc = rotated_test_vector(params_, bara);
    for (uint32_t i = 0; i < n; i++) {
        if (bara = mod_switch_from_torus32(2 * N, tlwe_[i]); bara == 0)
            continue;
        trlwe_lvl1 tmp0 = acc.polynomial_mul_by_xai_minus_one(bara);
        trlwe_lvl1 tmp1 = tmp0.trgswfft_external_product(bkfftlvl01[i]);
        for (uint32_t i = 0; i < N; i++) {
            acc.poly0()[i] += tmp1.poly0()[i];
            acc.poly1()[i] += tmp1.poly1()[i];
        }
    }

    return acc.sample_extract_index(0);
}

tlwe_lvl1 trlwe_lvl1::sample_extract_index(uint32_t index) const
{
    uint32vec tlwe(params_.N + 1);

    for (uint32_t i = 0; i <= index; i++)
        tlwe[i] = poly0_[index - i];
    for (uint32_t i = index + 1; i < params_.N; i++)
        tlwe[i] = -poly0_[params_.N + index - i];
    tlwe[params_.N] = poly1_[index];

    return tlwe_lvl1{params_, tlwe};
}

template <class RandomEngine>
key_switching_key::key_switching_key(const lwe_params& params,
                                     RandomEngine& randeng,
                                     const uint32vec& keylvl0,
                                     const uint32vec& keylvl1)
    : data_(params.N, params.t, (1 << params.basebit) - 1, params.n + 1)
{
    for (uint32_t i = 0; i < params.N; i++)
        for (uint32_t j = 0; j < params.t; j++)
            for (uint32_t k = 0; k < (1 << params.basebit) - 1; k++)
                data_[i][j][k] =
                    tlwe_lvl0::sym_encrypt(
                        params, keylvl0, randeng,
                        keylvl1[i] * (k + 1) *
                            (1U << (32 - (j + 1) * params.basebit)))
                        .get();
}

tlwe_lvl0 tlwe_lvl0::nand(const tlwe_lvl0& rhs,
                          const std::vector<trgswfft_lvl1>& bkfftlvl01,
                          const key_switching_key& ksk) const
{
    uint32vec ret(params_.n + 1);

    for (uint32_t i = 0; i <= params_.n; i++)
        ret[i] = -tlwe_[i] - rhs.tlwe_[i];
    ret[params_.n] += 1U << 29;

    return tlwe_lvl0{params_, ret}.gate_bootstrapping(bkfftlvl01, ksk);
}

}  // namespace detail
}  // namespace native
}  // namespace aqtfhe

#endif
