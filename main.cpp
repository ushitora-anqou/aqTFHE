#include "aqtfhe.hpp"

//

#include <gperftools/profiler.h>

#include <chrono>
#include <iostream>

//
#include "test.inc"

int main()
{
    test_aqtfhe();

    using namespace aqtfhe::native;

    std::random_device randeng;
    std::bernoulli_distribution binary(0.5);

    secret_key skey{lwe_params::tfhe_default(), randeng};
    cloud_key ckey = skey.get_cloud_key(randeng);

    const int TEST_SIZE = 100;

    // Prepare plain data
    std::vector<bool> p, q, r;
    for (int i = 0; i < TEST_SIZE; i++)
        p.push_back(binary(randeng));
    for (int i = 0; i < TEST_SIZE; i++)
        q.push_back(binary(randeng));
    for (int i = 0; i < TEST_SIZE; i++)
        r.push_back(!(p[i] & q[i]));

    // Encrypt the data
    std::vector<encrypted_bit> p_enc, q_enc, r_enc;
    for (int i = 0; i < TEST_SIZE; i++)
        p_enc.emplace_back(skey, randeng, p[i]);
    for (int i = 0; i < TEST_SIZE; i++)
        q_enc.emplace_back(skey, randeng, q[i]);
    r_enc.reserve(TEST_SIZE);

    // Calc NAND
    auto begin = std::chrono::high_resolution_clock::now();
    // ProfilerStart("sample.prof");
    for (int i = 0; i < TEST_SIZE; i++)
        r_enc.push_back(p_enc[i].nand(q_enc[i], ckey));
    // ProfilerStop();
    auto end = std::chrono::high_resolution_clock::now();

    // Print elapsed time.
    auto usec =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
    std::cerr << usec.count() / TEST_SIZE << "us / gate" << std::endl;

    // Check the results
    for (int i = 0; i < TEST_SIZE; i++)
        assert(r[i] == r_enc[i].decrypt(skey));

    // std::random_device randeng;
    // secret_key skey{lwe_params::tfhe_default(), randeng};

    // for (int i = 0; i < 10; i++) {
    //    assert(encrypted_bit(skey, randeng, true).decrypt(skey) == true);
    //    assert(encrypted_bit(skey, randeng, false).decrypt(skey) == false);
    //}

    // encrypted_bit bit0{skey, randeng, true};
    // encrypted_bit bit1{skey, randeng, false};
    // encrypted_bit bit2 = bit0.nand(bit1, skey.get_cloud_key(), randeng);
}
