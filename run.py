# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from app import app, db
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050,debug=True)