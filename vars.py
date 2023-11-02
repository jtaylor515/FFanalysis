# vars.py

"""This module defines project-level variables"""

gpu = True
positions = ["qb"]

# Positions Specific Prediction Variables and Variables to Drop
qb_predict = ['PASSING CMP', 'PASSING ATT', 'PASSING PCT', 'PASSING YDS', 'PASSING Y/A', 'PASSING TD', 'PASSING INT', 
'PASSING SACKS', 'RUSHING ATT', 'RUSHING YDS', 'RUSHING TD', 'MISC FL', 'MISC FPTS', 'WEEK']
qb_featureselect = ['POS RANK', 'POS', 'MISC G', 'MISC ROST', 'MISC FPTS/G', 'RECEIVING REC', 'RECEIVING TGT', 'RECEIVING YDS', 'RECEIVING Y/R',
'RECEIVING LG', 'RECEIVING 20+', 'RECEIVING TD', 'RUSHING Y/A', 'RUSHING LG', 'RUSHING 20+']