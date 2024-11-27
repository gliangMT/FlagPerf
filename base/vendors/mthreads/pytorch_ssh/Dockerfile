FROM sh-harbor.mthreads.com/ai_devtech/flagperf-base:torch_musa1.3.0-s4000-py38-v0.2
ENV MCCL_DEBUG=WARN
ENV MCCL_PROTOS=2

RUN apt update && \
    apt install -y openssh-server && \
    mkdir /var/run/sshd && \
    echo 'root:password' | chpasswd && \
    mkdir -p /root/.ssh && chmod 700 /root/.ssh

COPY id_rsa.pub /root/.ssh/authorized_keys
COPY id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/authorized_keys

RUN printf "Host *\n  StrictHostKeyChecking no\n User root\n SendEnv PATH\n  SendEnv LD_LIBRARY_PATH\n  SendEnv OPAL_PREFIX\n  Port 60128" > /root/.ssh/config && chmod 600 /root/.ssh/config && \
    echo 'Port 60128' >> /etc/ssh/sshd_config && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config && \
    echo 'AcceptEnv PATH LD_LIBRARY_PATH OPAL_PREFIX' >> /etc/ssh/sshd_config

EXPOSE 60128

CMD ["/usr/sbin/sshd", "-D"]