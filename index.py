#!/usr/bin/env python

import os
import sys
import site

appdir = os.path.dirname(__file__)
os.chdir(appdir)

site.addsitedir(appdir + '/lib/python3.10/site-packages')
site.addsitedir(appdir + '/lib64/python3.10/site-packages')

sys.path.insert(0, appdir)

prefixstart = len('/var/www/html');
os.environ["DASH_REQUESTS_PATHNAME_PREFIX"] = appdir[prefixstart:] + '/';

from app import server as application

