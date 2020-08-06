FROM gitpod/workspace-full

# Install custom tools, runtimes, etc.
# For example "bastet", a command-line tetris clone:
# RUN brew install bastet
#
# More information: https://www.gitpod.io/docs/config-docker/
RUN pyenv install 3.7.7 && pyenv global 3.7.7
RUN pip install "gin-config>=0.3.0" "paddlepaddle==1.8.3"