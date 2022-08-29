# Dashboard for exploring ECG data on Physionet

This repository contains code to genereate an interactive dashboard to view ECG data from the [Icentia11k](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) database on Physionet. It accompanies the Medium article published [here](https://medium.com/p/c748588e2920/edit).

## Running the app locally

To run the app on your local computer, clone the repository, create a virtual environment and install the necessary requirements with

`pip install --upgrade pip`
`pip install -r requirements`

Then run the app with

`python app.py`

## Medium files

To follow along with the Medium article, there are several versions of the app at different stages of completion in the [medium](./medium) directory. These can be run with e.g.

`cd medium`
`python app_v1.py`

If you cannot access the Medium article, there is pdf version [here](./medium/article.pdf).
