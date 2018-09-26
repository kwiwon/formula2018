# model from colab need python:3.6.3
FROM python:3.6 

# Install necessary packages
RUN apt-get update
RUN apt-get install -y python-opencv cppad coinor-libipopt-dev libuv1-dev libssl-dev gcc g++ cmake make

# cleanup 
RUN apt-get clean

# Create working directory
RUN mkdir /app

# install python packages
COPY formula-trend/requirements.txt /app
RUN pip install -r /app/requirements.txt

# Install uWebSockets
COPY formula-trend/mpc /app/mpc
RUN /app/mpc/install-ubuntu.sh
RUN ln -s /usr/lib64/libuWS.so /usr/lib/libuWS.so
  
# Copy the current directory contents into the container at /app
COPY formula-trend/ /app

# Set the working directory to /app
WORKDIR /app

# Run bot.py when the container launches, you should replace it with your program
# -b specify the bot type, "bc", "pid", "mpc"
ENTRYPOINT ["python3", "-u", "bot.py", "-b", "bc"]
