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

* 2021/10/09 Updated
* Utils.py
* CopyLastSteps.py
* HistoricalAverage.py
* STGCN.py
* GraphWaveNet.py
* DCRNN.py
* TGCN.py
* LSTNet.py
* MTGNN.py
* AGCRN.py
* GMAN.py

* traintest_STGCN.py
* traintest_GraphWaveNet.py
* traintest_TGCN.py
* traintest_DCRNN.py
* traintest_GMAN.py
* traintest_AGCRN.py
* traintest_MTGNN.py
* traintest_LSTNet.py
