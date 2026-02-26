# docker build -f Dockerfile -t than6785/zd:dreamer1 .

# docker run -it --gpus all --rm -v ~/logdir/docker:/logdir than6785/zd:dreamer1 python dreamerv3/main.py --logdir /logdir/{timestamp} --configs dmc_vision size1m 

FROM than6785/dreamer-base

# Envs
##RUN wget -O - https://gist.githubusercontent.com/danijar/ca6ab917188d2e081a8253b3ca5c36d3/raw/install-dmlab.sh | sh
##RUN pip install ale_py==0.9.0 autorom[accept-rom-license]==0.6.1
##RUN pip install procgen_mirror
##RUN pip install crafter
#RUN pip install dm_control
##RUN pip install memory_maze
##ENV MUJOCO_GL=egl
##RUN apt-get update && apt-get install -y openjdk-8-jdk && apt-get clean
##RUN pip install https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
##RUN chown -R 1000:root /venv/lib/python3.11/site-packages/minerl

# Source
RUN mkdir /app
WORKDIR /app
COPY . .
RUN chown -R 1000:root .

ENTRYPOINT ["sh", "entrypoint.sh"]