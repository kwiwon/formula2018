FROM centos:7.5.1804   
# Install necessary packages  
RUN yum install -y epel-release 
RUN yum -y install https://centos7.iuscommunity.org/ius-release.rpm 
RUN yum -y install python36u 
RUN yum -y install python36u-pip       
RUN yum -y install python36u-devel       
RUN yum -y install opencv-python       
RUN yum -y install libSM       
RUN yum install -y gcc       
RUN pip3.6 install -U pillow flask-socketio eventlet numpy scipy opencv-python
RUN pip3.6 install pillow
RUN pip3.6 install flask-socketio
RUN pip3.6 install opencv-python
RUN pip3.6 install eventlet
RUN pip3.6 install scipy
RUN pip3.6 install tensorflow==1.10.1
RUN pip3.6 install keras==2.2.2
RUN pip3.6 install pandas
RUN pip3.6 install scikit-learn
RUN pip3.6 install matplotlib
RUN yum clean all       
RUN rm -rf /root/.cache       
RUN mkdir /app
# Set the working directory to /app  
WORKDIR /app     
ARG arg_model  
ENV FORMULAR_MODEL=$arg_model  
ADD ./ /app  
#ENTRYPOINT python3.6 -m trend_formula $FORMULAR_MODEL -m drive
ENTRYPOINT ["python3.6", "drive.py"]
