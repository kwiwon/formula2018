FROM centos:7
  
# Install necessary packages
RUN yum install -y epel-release
RUN yum install -y libSM
RUN yum install -y python-devel
RUN yum install -y python-pip
RUN yum install -y opencv-python
RUN yum install -y gcc
  
# Copy the current directory contents into the container at /app
RUN mkdir /app
ADD . /app
  
# Set the working directory to /app
WORKDIR /app

# Install required python package
RUN pip install -r /app/requirements.txt

# Using Tensorflow as Keras backend
ENV KERAS_BACKEND tensorflow

ENTRYPOINT ["python", "formula_bot_w2_QT.py"]
