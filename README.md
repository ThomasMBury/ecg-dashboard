# Dashboard for exploring ECG data on Physionet

This repository contains code to genereate an interactive dashboard to view ECG data from the [Icentia11k](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) database on Physionet. It accompanies the Medium article published [here](https://medium.com/p/c748588e2920/edit).

## Running the app locally

To run the app on your local computer, clone the repository, create a virtual environment and install the necessary requirements with
```
pip install --upgrade pip
pip install -r requirements.txt
```

Then run the app with
```
python app.py
```

This should show
```
Dash is running on http://127.0.0.1:8050/
```

You can now visit the app at this URL. 


## Medium files

To follow along with the Medium article, there are several versions of the app at different stages of completion in the [medium](./medium) directory. These can be run with e.g.

```
cd medium
python app_v1.py
```

If you cannot access the Medium article, there is pdf version [here](./medium/article.pdf).

## Usage

Feel free to use and extend this template for your own use. If you find it useful, please consider 'clapping' the [medium article](https://medium.com/p/c748588e2920/edit) and starring this repository. Thank you!
