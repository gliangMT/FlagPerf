mcc gemm.mu -lmusart -I$MUTLASS_PATH/include -I$MUTLASS_PATH/tools/util/include/ -o gemm --offload-arch=mp_22 --std=c++17 -Ofast
./gemm