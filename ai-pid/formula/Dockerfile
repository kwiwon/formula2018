# Use an official Python runtime as a parent image
FROM python:3.6
 
# Set the working directory to /app
WORKDIR /app
 
# Install the python package we need
ADD requirements.txt /app/
RUN pip install -r /app/requirements.txt; \
    mkdir /log
 
# Copy the current directory contents into the container at /app
ADD ai_bot.py /app/
COPY sign-02 /app/sign-02
COPY sign-03 /app/sign-03
COPY sign-04 /app/sign-04
COPY sign-05 /app/sign-05
COPY sign-07 /app/sign-07
COPY sign-08 /app/sign-08
ADD utils_json.py /app/
ADD model.h5 /app/model.h5

 
# Run sample_bot.py when the container launches, you should replace it with your program
ENTRYPOINT ["python", "ai_bot.py", "-nd", "--model", "model.h5"]
