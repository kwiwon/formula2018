# 2018 Global AI Contest


## When
- Registration: 20:00, Aug. 24 – 23:59, Sep. 7, 2018
- Public beta: Sep. 3 – Sep. 13, 2018
- Preliminary Competition: Sep. 18 – Oct. 26, 2018
- Announcement of Top 300: Oct. 29, 2018
- **Championship Competition in Fukuoka, Japan:**
  - Elimination Rounds: Dec. 3, 2018
  - Final Round: Dec. 4, 2018

_Note: All times are UTC+9_


## What
- [Trend Hearts](https://wiki.jarvis.trendmicro.com/x/y3LXDg)
- [Formula Trend](https://wiki.jarvis.trendmicro.com/x/AnnXDg)


## How

The relevant training materials and SDK information can be found in the following places. Of course, the sample codes are ready for newbies to get familiar with both competition games.

- [Official game portal](http://aicontest2018.trendmicro.com) (Open Beta on Sep. 3)
- [Training materials](https://wiki.jarvis.trendmicro.com/x/uC_uD)
- [SDK information](https://wiki.jarvis.trendmicro.com/x/uG3XDg)
- [Game rules](https://wiki.jarvis.trendmicro.com/x/Y23XDg)
- [Submit a bug](http://aicontest2018.trendmicro.com/)


## Issue Mitigation

#### Formula Trend

- Failed to record training data on MacOS

  **[Mitigation]** Please move the application “formula-trend-1.0.0-alpha.3” to the Application folder on Mac.

- Failed to run sample_bot.py due to socket.io error on MacOS or Windows

  **[Mitigation]** Please upgrade the version of pip and flask-socketio module using the commands below:

      $ pip install --upgrade pip 
      $ sudo pip install flask-socketio==1.0b1 --user flas