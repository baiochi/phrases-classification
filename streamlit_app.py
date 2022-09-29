
import streamlit as st
import numpy as np
import pandas as pd
import joblib

from utils import select_model, compute_metrics

from sklearn import set_config
set_config(display="diagram")

LABELS = np.array(['educação', 'finanças', 'indústrias', 'orgão público', 'varejo'])
df = pd.read_csv('data/dataset.csv')

st.set_page_config(
    page_title="Hand Talk Challenge",
    page_icon="style/favicon.png",
)

if 'model' not in st.session_state:
    st.session_state['model'] = False

st.title('Multi-label Classification of Phrases')
st.header('Hand Talk Challenge')
st.markdown(
	'''
	Write your own sentence and see how the model classifies it.
	You can also select different models and see how they perform.
	'''
)

# --------------------------------------------------------------------
# 								Sidebar		
# --------------------------------------------------------------------

model_selected = st.sidebar.selectbox(
	label="Select the model:", 
	options=[
		# "Decision Tree",
		# "Gradient Boosting",
		# "Logistic Regression",
		"Multinomial Naive Bayes",
		"Optimized Naive Bayes",
		# "Random Forest", 
		# "Support Vector Machine",
		])

if model_selected:
    st.session_state["model"] = select_model(model_selected)
    st.write(st.session_state["model"])
    model_metrics = compute_metrics(df, st.session_state["model"])
    
# metrics_df, train_roc_auc_fig, test_roc_auc_fig,
# clf_report_train, clf_report_test

# ---------------------------------------------------------------

st.sidebar.write('Pipeline diagram: ')
st.sidebar.write(st.session_state['model'][1])

# --------------------------------------------------------------------
# 							Main Section		
# --------------------------------------------------------------------

# Given a input sentence, predict the labels
with st.form(key='my_form'):
	text_input = st.text_input(label="Write your sentence here:")
	submit_button = st.form_submit_button(label='Predict')

	if submit_button:
		# Labels predicted
		labels_pred = st.session_state["model"].predict([text_input])[0].astype(bool)
		labels_pred = LABELS[labels_pred]
		if labels_pred.size==0:
			st.write("No labels predicted.")
		else:
			st.write(f'Predicted labels: {labels_pred}')
		
		# Labels probabilities
		pred_proba = st.session_state["model"].predict_proba([text_input])

		if len(pred_proba) == 1:	# ClassifierChains case
			pred_proba_df = pd.DataFrame({
				'Category': LABELS,
				'Probability': [str(round(proba*100,2)) + ' %' for proba in pred_proba[0]]
			})
		else:		# Other cases
			pred_proba_df = pd.DataFrame({
				'Category': LABELS,
				'Probability': [str(round(proba[0][1]*100,2)) + ' %' for proba in pred_proba]
			})
		st.write('Predicted probabilities:')
		st.dataframe(pred_proba_df)

# View data frame
if st.checkbox('Preview data frame', value=False, key='df_checkbox'):
	st.dataframe(df)

# Classification Report
if st.checkbox('Show Classification Report', value=False, key='clf_report_checkbox'):
	col1, col2 = st.columns(2)
	with col1:
		st.subheader('Train dataset')
		st.dataframe(model_metrics['clf_report_train'])
	with col2:
		st.subheader('Test dataset')
		st.dataframe(model_metrics['clf_report_test'])

# ROC AUC
if st.checkbox('Show ROC AUC', value=False, key='roc_auc_checkbox'):
	col1, col2 = st.columns(2)
	with col1:
		st.subheader('Train dataset')
		st.plotly_chart(model_metrics['train_roc_auc_fig'])
	with col2:
		st.subheader('Test dataset')
		st.plotly_chart(model_metrics['test_roc_auc_fig'])

st.markdown('author: [@baiochi](http://github.com/baiochi)')



