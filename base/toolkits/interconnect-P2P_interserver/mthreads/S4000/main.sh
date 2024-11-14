#!/bin/bash
service ssh restart
export NCCL_DEBUG=WARN
# export OPAL_PREFIX=/opt/hpcx/ompi
# export PATH=/usr/local/nvm/versions/node/v16.20.2/bin:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin
# export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
mcc -c -o bandwidth.o bandwidth.mu -I/usr/local/musa/include -I/usr/lib/x86_64-linux-gnu/mpich/include -fPIE -fPIC
mpic++ -o bdtest bandwidth.o -L/usr/local/musa/lib -lmusart -lmccl -lmusa -lmpi
HOSTS=$(yq '.HOSTS | join(",")' ../../../../configs/host.yaml)
echo "NODERANK: $NODERANK"
if [ "$NODERANK" -eq 0 ]; then
    echo "NODERANK is 0, executing the final command..."
    mpirun --allow-run-as-root --host $HOSTS -np 2 -x NCCL_DEBUG=WARN -x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5_2,mlx5_5 -x CUDA_DEVICE_MAX_CONNECTIONS=1 ./bdtest
fi