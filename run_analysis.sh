#!/usr/bin/env bash
mkdir -p data
mkdir -p figs
cwd=$(pwd)
python presidents-lsa/scrape_data.py ${cwd}/data
python presidents-lsa/PresidentsLSA.py 10 ${cwd}/data/presidents.csv ${cwd}/data/ideology_queries.csv ${cwd}/figs >> query_output.txt