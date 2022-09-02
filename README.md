# Building a dashboard in Plotly Dash

Generate an interactive dashboard to explore ECG data from the [Icentia11k](https://physionet.org/content/icentia11k-continuous-ecg/1.0/) database on Physionet. It accompanies this [Medium article](https://medium.com/p/c748588e2920/edit). The app is currently hosted on Heroku [here](https://ecg-dashboard-medium.herokuapp.com/).


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

## Usage

Feel free to use and extend this template for your own use. If you find it useful, please consider 'clapping' the [Medium article](https://medium.com/p/c748588e2920/edit) and starring this repository. Thank you!

## Preview

<img width="1164" alt="app" src="https://user-images.githubusercontent.com/36854425/187282440-138884a1-f473-4f8d-a8ff-54e1f197f31b.png">

## Citations

Tan, S., Ortiz-Gagné, S., Beaudoin-Gagnon, N., Fecteau, P., Courville, A., Bengio, Y., & Cohen, J. P. (2022). Icentia11k Single Lead Continuous Raw Electrocardiogram Dataset (version 1.0). PhysioNet. https://doi.org/10.13026/kk0v-r952.

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

