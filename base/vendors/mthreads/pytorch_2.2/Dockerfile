FROM registry.mthreads.com/mcconline/musa-pytorch-release-public:rc3.1.0-v1.3.0-S4000-py38
# RUN /bin/bash -c "pip config set global.index-url https://mirror.baidu.com/pypi/simple"
RUN /bin/bash -c "uname -a"
RUN /bin/bash -c alias python3=python
ENV PATH="/opt/conda/envs/py38/bin:$PATH"
RUN apt-get update
RUN pip3 install loguru
RUN pip3 install schedule
RUN pip3 install munch
RUN pip3 install argparse
RUN pip3 install pyyaml
ENTRYPOINT []
FROM sh-harbor.mthreads.com/ai_devtech/flagperf-base:torch_musa1.3.0-s4000-py38-v0.2
RUN apt update && apt install ipmitool -y