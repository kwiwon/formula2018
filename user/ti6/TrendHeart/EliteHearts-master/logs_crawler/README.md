Log Crawler
===========
Crawl all game logs from the [official contest website](https://aicontest2018.trendmicro.com/#/trend-hearts/ranked-game-history)

Usage
-----
1. Python 3.6+ is required
2. Run `python3 -m pip install -r requirements.txt`
3. Run `python3 crawl_logs.py N` and login with your domain account
  - Please use the maximum id of fetched logs as `N` (e.g., 2000)
4. All logs will be stored into `logs/`
