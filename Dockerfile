# docker build -f Dockerfile -t than6785/dreamer .
# docker push than6785/dreamer

# docker run -it --gpus all --rm -v ~/logdir/docker:/logdir than6785/dreamer python dreamerv3/main.py --logdir /logdir/{timestamp} --configs dmc_vision size1m 

FROM than6785/dreamer-base

# Envs (base already has ale_py/autorom + dm_control; crafter needed for Tier 1+)
RUN pip install crafter
##RUN wget -O - https://gist.githubusercontent.com/danijar/ca6ab917188d2e081a8253b3ca5c36d3/raw/install-dmlab.sh | sh
##RUN pip install ale_py==0.9.0 autorom[accept-rom-license]==0.6.1
##RUN pip install procgen_mirror
#RUN pip install dm_control
##RUN pip install memory_maze
##ENV MUJOCO_GL=egl
##RUN apt-get update && apt-get install -y openjdk-8-jdk && apt-get clean
##RUN pip install https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
##RUN chown -R 1000:root /venv/lib/python3.11/site-packages/minerl

# Vast.ai SSH Flakiness Fix:
# Vast's internal `/.launch` script tries to install openssh-server on the fly via `apt-get` during container boot.
# This frequently fails or times out when multiple instances are launched or Ubuntu mirrors are slow, causing
# the SSH daemon to never start (even when the container and your code are running perfectly fine).
# By baking openssh-server into our image here, Vast's launch script will successfully find and start the daemon
# even if its own `apt-get` update step fails.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends openssh-server tmux rsync curl less && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Tailscale
RUN curl -fsSL https://tailscale.com/install.sh | sh

# Source
RUN mkdir /app
WORKDIR /app
COPY . .
RUN chown -R 1000:root .

ENTRYPOINT ["sh", "entrypoint.sh"]