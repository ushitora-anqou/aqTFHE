#include "aqtfhe.hpp"

////
//#include <cereal/archives/json.hpp>
//#include <cereal/types/array.hpp>
//#include <fstream>
//
// struct cloud_key_export {
//    std::array<uint32_t, 500 + 1> tlwe;
//    std::array<
//        std::array<std::array<std::array<uint32_t, 500 + 1>, (1 << 2) - 1>,
//        8>, 1 << 10> ksk;
//    std::array<std::array<std::array<std::array<double, 1 << 10>, 2>, 2 * 2>,
//               500>
//        bkfftlvl01;
//    std::array<uint32_t, 500 + 1> bootedtlwe;
//
//    template <class Archive>
//    void serialize(Archive &ar)
//    {
//        ar(tlwe, ksk, bkfftlvl01, bootedtlwe);
//    }
//};

void test_aqtfhe()
{
    using namespace aqtfhe::native;
    std::random_device randeng;

    const lwe_params &params = lwe_params::tfhe_default();
    secret_key skey{params, randeng};
    cloud_key ckey = skey.get_cloud_key(randeng);

    {
        // Test encrypting/decrypting a bit.
        for (int i = 0; i < 10; i++) {
            assert(encrypted_bit(skey, randeng, true).decrypt(skey) == true);
            assert(encrypted_bit(skey, randeng, false).decrypt(skey) == false);
        }
    }

    {
        // Test trlwe_lvl1::sym_encrypt/sym_decrypt.
        std::bernoulli_distribution binary(0.5);

        for (int i = 0; i < 10; i++) {
            uint32vec p(params.N);
            for (uint32_t &val : p)
                val = binary(randeng) > 0;

            uint32vec pmu(params.N);
            for (uint32_t i = 0; i < params.N; i++)
                pmu[i] = p[i] ? params.mu : -params.mu;

            detail::trlwe_lvl1 encrypted = detail::trlwe_lvl1::sym_encrypt(
                params, skey.keylvl1(), randeng, pmu);
            uint32vec res = encrypted.sym_decrypt(skey.keylvl1());

            assert(p == res);
        }
    }

    {
        // Test trlwe_lvl1::trgswfft_external_product
        std::bernoulli_distribution binary(0.5);

        for (int i = 0; i < 10; i++) {
            uint32vec p(params.N);
            // for (uint32_t &val : p)
            //    val = binary(randeng) > 0;
            for (auto &val : p)
                val = false;

            uint32vec pmu(params.N);
            for (uint32_t i = 0; i < params.N; i++)
                pmu[i] = p[i] ? params.mu : -params.mu;

            detail::trlwe_lvl1 trlwe = detail::trlwe_lvl1::sym_encrypt(
                params, skey.keylvl1(), randeng, pmu);

            detail::trgswfft_lvl1 trgswfft = detail::trgswfft_lvl1::sym_encrypt(
                params, skey.keylvl1(), randeng, 1);

            uint32vec res =
                trlwe.trgswfft_external_product(trgswfft).sym_decrypt(
                    skey.keylvl1());

            assert(p == res);
        }
    }

    {
        // Test tlwe_lvl0::gate_bootstrapping
        std::bernoulli_distribution binary(0.5);

        for (int i = 0; i < 10; i++) {
            bool p = binary(randeng) > 0;
            detail::tlwe_lvl0 tlwe = detail::tlwe_lvl0::sym_encrypt(
                params, skey.keylvl0(), randeng, p ? params.mu : -params.mu);
            detail::tlwe_lvl0 booted_tlwe =
                tlwe.gate_bootstrapping(ckey.bkfftlvl01(), ckey.ksk());
            bool p2 = booted_tlwe.sym_decrypt(skey.keylvl0());
            assert(p == p2);
        }
    }

    {
        // Test tlwe_lvl0::nand
        std::bernoulli_distribution binary(0.5);

        for (int i = 0; i < 10; i++) {
            bool p = binary(randeng) > 0, q = binary(randeng);

            detail::tlwe_lvl0 p_enc = detail::tlwe_lvl0::boots_sym_encrypt(
                                  params, randeng, skey.keylvl0(), p),
                              q_enc = detail::tlwe_lvl0::boots_sym_encrypt(
                                  params, randeng, skey.keylvl0(), q);

            detail::tlwe_lvl0 res_enc =
                p_enc.nand(q_enc, ckey.bkfftlvl01(), ckey.ksk());

            bool res = res_enc.boots_sym_decrypt(skey.keylvl0());
            assert(res == !(p & q));
        }
    }

    /*
{
    std::ifstream ifs("out.cereal", std::ios::binary);
    cereal::JSONInputArchive archive(ifs);
    cloud_key_export export_data;
    archive(export_data);

    const lwe_params &params = lwe_params::tfhe_default();

    auto ksk_src = make_nested_vector<uint32_t>(
        params.N, params.t, (1 << params.basebit) - 1, params.n + 1);
    for (int i = 0; i < params.N; i++)
        for (int j = 0; j < params.t; j++)
            for (int k = 0; k < (1 << params.basebit) - 1; k++)
                for (int l = 0; l < params.n + 1; l++)
                    ksk_src[i][j][k][l] = export_data.ksk[i][j][k][l];
    detail::key_switching_key ksk{ksk_src};

    std::vector<detail::trgswfft_lvl1> bkfftlvl01;
    for (size_t i = 0; i < params.n; i++) {
        auto data = make_nested_vector<double>(2 * params.l, 2, params.N);
        for (size_t j = 0; j < 2 * params.l; j++) {
            for (size_t k = 0; k < params.N; k++) {
                data[j][0][k] = export_data.bkfftlvl01[i][j][0][k];
                data[j][1][k] = export_data.bkfftlvl01[i][j][1][k];
            }
        }
        bkfftlvl01.emplace_back(params, data);
    }

    detail::tlwe_lvl0 tlwe = detail::tlwe_lvl0{
        params,
        uint32vec(export_data.tlwe.begin(), export_data.tlwe.end())};

    detail::tlwe_lvl0 booted_tlwe =
        tlwe.gate_bootstrapping(bkfftlvl01, ksk);

    assert(0);  // Check booted_tlwe and export_data.bootedtlwe
}
*/
}
