#!/bin/bash
mcc gemm.mu -o gemm -lmusart --offload-arch=mp_22 
./gemm