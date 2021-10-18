# TwitterMobility1 [Another Repository for developing TwitterMobility]
* Started to use ConfigParser to read config file (params.txt). 
* Started to use logging (logging.info) instead of print() to simultaneously save the "print" result to logging.txt file.
* Now you only need to specify the "ex" argument in cmd line.

<br>

* cd model
* python traintest_MODELNAME.py --ex=EXPERIMENT --gpu=GPU_ID
* EXPERIMENT = {typhoon-inflow, typhoon-outflow, covid-inflow, covid-outflow}

* cd model+tw
* python traintest_MODELNAME.py --ex=EXPERIMENT --gpu=GPU_ID
* EXPERIMENT = {typhoon-inflow, typhoon-outflow, covid-inflow, covid-outflow}

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
* traintest_MTGNN.py
* traintest_AGCRN.py
* traintest_TransformerT.py

