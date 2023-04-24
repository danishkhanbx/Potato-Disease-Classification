import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from flask import Flask
from .config import Config

app = Flask(__name__)
app.config.from_object(Config)

from .views import views
app.register_blueprint(views)
