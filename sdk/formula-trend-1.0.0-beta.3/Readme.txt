Formula Trend Training SDK

## Requirement and Tested Environment
bot:
-Python 2.7
-pip install pillow
-pip install flask-socketio
-pip install opencv-python
-pip install eventlet

Simulator:
Windows/Ubuntu 16.04/Mac

## How to execute
1) Enable the car simulator first by 
    - double click
    - command ./formula-trend-1.0.0-alpha.4.x64.exe --ip 127.0.0.1 --port 4567 [--track <track_no>]
2) Follow the sdk introduction (https://wiki.jarvis.trendmicro.com/display/2SC/Formula+Trend+SDK) to 
    - record training data or 
    - run in autonomous mode to connect with bot
3) Run sample_bot.py

## How to run simulator by creating virtual frame buffer (if you do not have default X display to render out)
apt-get install xvfb
Xvfb :10 -ac -screen 0 1024x768x24 &
export DISPLAY=:10
then run the simulator as usual 

## How to run simulator without rendering the third person view to the display 
add -batchmode (single dashline, it is supported by unity)
note you still need a default display to render the 'car camera image' for bot, 
to assign default display, set DISPLAY variable before run a simulator (normally, the command is 'export DISPLAY=:0')

## Why I always crash when running the simulator
Currenly we found unity crash directly if the display can not be found.
if you already check the dislay problem in above howtos, please contact the taskforce.

## Telemetry sending time
Simulator always sends one telemetry on receiving one command. 
if command received, but the car controlling has not updated. It still sends the telemetry out.

## Other support I can get
- Q&A session in the buttom of SDK wiki page https://wiki.jarvis.trendmicro.com/display/2SC/Formula+Trend+SDK (faster)
- Submit a bug report by https://aicontest2018.trendmicro.com/

## Simulator resolution out of my screen
To fix this, locate the following registry key in windows:  HKCU\Software\Software\Trend\Formula Trend and delete the entire key, then run the game.
In Mac, delete the corresponding preferences file located in ~/Library/Preferences/unity

