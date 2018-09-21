# model from colab need python:3.6.3
FROM python:3.6 

# Install necessary packages
RUN apt-get update
RUN apt-get install -y python-opencv

# install python packages   
# RUN pip install pillow flask-socketio eventlet tensorflow==1.10.1 keras numpy==1.14.5

# cleanup 
RUN apt-get clean
  
# Copy the current directory contents into the container at /app
RUN mkdir /app
COPY bot_candidates/formula-trend/05/ /app

# install python packages 
RUN pip install -r /app/requirements.txt

# Set the working directory to /app
WORKDIR /app

# Run bot.py when the container launches, you should replace it with your program
# -b specify the bot type, "bc", "pid", "mpc"
ENTRYPOINT ["python3", "bot.py", "-b", "bc"]
