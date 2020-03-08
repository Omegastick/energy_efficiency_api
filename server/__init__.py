from flask import Flask

from server.config import create_app, setup_logging

setup_logging()
app = create_app(__name__)

from server import views
