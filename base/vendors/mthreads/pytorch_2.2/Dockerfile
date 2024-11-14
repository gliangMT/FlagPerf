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
# for mpi
# RUN echo "ulimit -n 65536" >> ~/.bashrc
# RUN echo "ulimit -l unlimited" >> ~/.bashrc
# RUN curl -L https://github.com/mikefarah/yq/releases/download/v4.44.3/yq_linux_amd64 -o /usr/bin/yq && chmod +x /usr/bin/yq
# RUN wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
# RUN tar zxvf openmpi-4.1.5.tar.gz
# RUN cd openmpi-4.1.5 && mkdir -p /opt/hpcx/ompi/ && ./configure --prefix=/opt/hpcx/ompi/
# RUN make && make install
