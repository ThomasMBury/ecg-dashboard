#! /usr/bin/python3.6

import logging
import sys
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/home/ubuntu/cloud_holter_app/app-holter-ubc-ecg')
sys.path.insert(0,'/home/ubuntu/cloud_holter_app/app-holter-ubc-ecg/venv/lib/python3.6/site-packages')
from app import server as application
application.secret_key = 'holter'
