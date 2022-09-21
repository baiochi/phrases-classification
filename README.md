# Phrases classification by sector

[Hand Talk](https://www.linkedin.com/company/hand-talk/) selection process technical challenge. 

## Description

Given a database with classified phrases in specific context/sectors (economical activities), your goal is to create a classification model to predict these phrases in the annotatted sectors.

The following [dataset](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dbc8e17b-c43e-40e8-ae0f-9ab50d80aed4/dataset.csv) will be used to train your model. This data cointains **521** phrases classified among the sectors:  `finanças, educação, indústrias, varejo, orgão público`.

Some phrases are classified with more than one sector, this happens when the phrase has vocabularies with terms from different sectors. 
Example:  

> **Curso de Técnico em Segurança do Trabalho por 32x R$ 161,03.**  

This phrase could be classified both in the sectors `educação` and `finanças`.

Therefore, your model must deal with the possibility of the sentence having multiple classifications.

## Bonus challenge

Deploy your model in a web application with *JavaScript*.  








