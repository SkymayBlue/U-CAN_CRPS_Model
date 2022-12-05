# UCAN CRPS Model
![python](https://img.shields.io/badge/Python%20tested-3.9.x%20-blue)
![tensorflow](https://img.shields.io/badge/tensorflow%20tested-2.9.x%20-blue)
![version](https://img.shields.io/badge/version-v.1.0.0-blue)

UCAN_CRPS_Model is written in the Python3 programming language. We use ResNet50 and Tensorflow-2.3 to trainied a classifier on UCAN's colorectal pathway profile. This repo used for predict CRPS (Colorectal Prognostic Subgroups) catrgory.


---

# _Documentation_

- [Installation](#installation)
  - [Download zip](#Download-zip)
  - [Clone repository](#Clone-repository)
-[Setting Enviroment](#Setting-Enviroment)
- [Run](#run)
- [Viewing the results](#viewing-the-results)

---
## Installation

### Download zip
```bash
wget https://github.com/SkymayBlue/UCAN_CRPS_Model/archive/master.zip
unzip UCAN_CRPS_Model-master.zip
```
or
### Clone repository
```bash
git clone https://github.com/SkymayBlue/UCAN_CRPS_Model.git
```

---

## Setting Environment
A typical user can install the libraries using the following command:
``` bash
python3 -m pip install -r requirements.txt
```

---

# Run
run main.py will open the browser with the url http://127.0.0.1:8080/, you need click the button "Try it out" in POST, then you upload and predict the csv file which in testdata directory and click Execute to test if your install corrected or not. 
```bash
python3 main.py
```
If you are succeed, you will see response like this.![response](img/img_2.png)

# Viewing the results
the "predictions_only.xlsx" in "predict" dir have eight columns: samples, the probability and the most likely class.![result](img/img_3.png)