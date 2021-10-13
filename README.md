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
* traintest_STGCN.py
* GraphWaveNet.py
* traintest_GraphWaveNet.py
* DCRNN.py
* traintest_DCRNN.py
* TGCN.py
* traintest_TGCN.py
* LSTNet.py
* traintest_LSTNet.py
* MTGNN.py
* traintest_MTGNN.py
* AGCRN.py
* traintest_AGCRN.py
* GMAN.py
* traintest_GMAN.py

