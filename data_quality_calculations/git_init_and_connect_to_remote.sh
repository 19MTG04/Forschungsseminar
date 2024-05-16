#!/bin/bash

# Before executing the script you need to create a git project with the name of the project slug (the project name with the dashes).

git init
git remote add origin https://github.com/gesc_ma/data_quality_calculations.git
git add .
git commit -m "Initial commit"
git push -u origin main
