# Use an official Python runtime as a parent image
FROM python:2.7-slim
 
# Set the working directory to /app
WORKDIR /app
 
# Install the python package we need
RUN pip install --trusted-host pypi.python.org websocket-client==0.7.0
 
# Copy the current directory contents into the container at /app
ADD . /app
 
# Run sample_bot.py when the container launches, you should replace it with your program
# The parameters of the program should be "[player_name] [player_number] [token] [connect_url]"
ENTRYPOINT ["python", "sample_bot.py"]
