# Install base image
FROM python:2.7

# Install necessary packages
RUN apt-get update
RUN apt-get install -y python-opencv cppad coinor-libipopt-dev

# cleanup
RUN apt-get clean

# Copy the current directory contents into the container at /app
RUN mkdir /app
COPY ./ /app

# Install uWebSockets
RUN /app/install-ubuntu.sh
RUN ln -s /usr/lib64/libuWS.so /usr/lib/libuWS.so

# install python packages 
RUN pip install -r /app/requirements.txt

# Set the working directory to /app
WORKDIR /app

# Run bot.py when the container launches, you should replace it with your program
ENTRYPOINT ["python", "-u", "sample_bot.py"]
