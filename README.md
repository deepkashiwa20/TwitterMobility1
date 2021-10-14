# TwitterMobility1 [Another Repository for developing TwitterMobility]
* Started to use ConfigParser to read config file (params.txt). 
* Started to use logging (logging.info) instead of print() to simultaneously save the "print" result to logging.txt file.
* Now you only need to specify the "ex" argument in cmd line.

<br>

* cd model
* python CopyLastSteps.py --ex=typhoon-inflow-kanto8
* python CopyLastSteps.py --ex=typhoon-outflow-kanto8
* python CopyLastSteps.py --ex=covid-inflow-kanto8
* python CopyLastSteps.py --ex=covid-outflow-kanto8

<br>

* CopyLastSteps.py
* CopyYesterday.py
* HistoricalAverage.py
* traintest_STGCN.py
* traintest_DCRNN.py
* traintest_GraphWaveNet.py
* traintest_ASTGCN.py
* traintest_TGCN.py
* traintest_LSTNet.py
* traintest_GMAN.py
* traintest_AGCRN.py
* traintest_MTGNN.py
