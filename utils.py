
# Utils for computing models in the Streamlit app

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import hamming_loss, auc, roc_curve, roc_auc_score, f1_score, classification_report


# Compute metrics for multiclass ROC AUC
def multiclass_rouc_auc(y_true, y_score, n_classes):
    
    # Get FP/TP and ROC AUC Score for each class
    fpr = dict(); tpr = dict(); roc_auc = dict();

    for i in range(y_score.shape[1]):
        # false positives and true positives
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        # roc auc score
        roc_auc[i] = roc_auc_score(y_true[:, i], y_score[:, i])
    
    # Compute micro-average ROC curve and ROC area from prediction scores
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Average it and compute AUC
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    
    return fpr, tpr, roc_auc

# Create figure of multiclass ROC AUC
def plot_multiclass_roc_auc(fpr, tpr, roc_auc, labels, title=None):
    
    fig = go.Figure()

    # Baseline roc curve
    fig.add_shape(type='line', line=dict(dash='dash'),
                x0=0, x1=1, y0=0, y1=1)
    # Plot curve for each clas
    for index, label in enumerate(labels):
        name = f'{label} (area = {roc_auc[index]:.3f})'
        fig.add_trace(go.Scatter(x=fpr[index], y=tpr[index], name=name, mode='lines'))

    # Plot micro and macro average
    fig.add_trace(go.Scatter(x=fpr['macro'], y=tpr['macro'], name=f'macro-average (area = {roc_auc["macro"]:.2f})', mode='lines', line_dash='dot'))
    fig.add_trace(go.Scatter(x=fpr['micro'], y=tpr['micro'], name=f'micro-average (area = {roc_auc["micro"]:.2f})', mode='lines', line_dash='dot'))

    fig.update_layout(
        title_text = title + ' ROC curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        xaxis=dict(constrain='domain'),
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.55
        ),
        autosize=False,
        width=600, height=500,
    )
    
    return fig



def compute_metrics(df, model):

	# Read data, set features and targets
	X = df['sentence']
	y = df['category']

	# Binarize labels
	mlb = MultiLabelBinarizer()
	y = mlb.fit_transform(df['category'].apply(str.split, sep=','))
	labels = mlb.classes_

	# Train test split
	X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=42)

	# Predictions
	y_pred_train = model.predict(X_train)
	y_pred_test = model.predict(X_test)
	y_proba_train = model.predict_proba(X_train)
	y_proba_test = model.predict_proba(X_test)

	# Acc score
	avr_acc_train = model.score(X_train, y_train)*100
	avr_acc_test = model.score(X_test, y_test)*100
	
	# Hamming Loss
	hamming_loss_train = hamming_loss(y_train, model.predict(X_train))
	hamming_loss_test = hamming_loss(y_test, model.predict(X_test))

	# Cross validation
	cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()*100

	# ROC AUC
	fpr_train, trp_train, roc_auc_train = multiclass_rouc_auc(y_train, y_proba_train, len(labels))
	fpr_test, trp_test, roc_auc_test = multiclass_rouc_auc(y_test, y_proba_test, len(labels))
	# ROC AUC figure
	train_roc_auc_fig = plot_multiclass_roc_auc(fpr_train, trp_train, roc_auc_train, labels, title='Train dataset')
	test_roc_auc_fig = plot_multiclass_roc_auc(fpr_test, trp_test, roc_auc_test, labels, title='Test dataset')

	# Compute precision, recall and f1 score for each label, plus micro and macro average
	clf_report_train = pd.DataFrame(classification_report(y_train, y_pred_train, output_dict=True)).T
	clf_report_test = pd.DataFrame(classification_report(y_test, y_pred_test, output_dict=True)).T

	# Rename axis with labels
	clf_report_train.index = list(labels) + ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
	clf_report_test.index = list(labels) + ['micro avg', 'macro avg', 'weighted avg', 'samples avg']

	# Add the ROC AUC scores computed previously
	clf_report_train['roc auc'] = list(roc_auc_train.values()) + ([np.nan] * 2)
	clf_report_test['roc auc'] = list(roc_auc_test.values()) + ([np.nan] * 2)
	
	metrics_df = pd.DataFrame({
		'Train': [avr_acc_train, hamming_loss_train],
		'Test': [avr_acc_test, hamming_loss_test],
	},index=['Accuracy', 'Hamming Loss'])
	print(f'Cross Validation Score: {cv_score:.2f} %')

	return {
		'metrics_df': metrics_df,
		'train_roc_auc_fig': train_roc_auc_fig,
		'test_roc_auc_fig': test_roc_auc_fig,
		'clf_report_train': clf_report_train,
		'clf_report_test': clf_report_test,
	}

# Streamlit render: select model from list
def select_model(model_selected):

	if model_selected == "Decision Tree":
		return joblib.load('models/decision_tree.joblib')

	elif model_selected=="Gradient Boosting":
		return joblib.load('models/gradient_boosting.joblib')

	elif model_selected=="Logistic Regression":
		return joblib.load('models/logistic_regression.joblib')

	elif model_selected=="Multinomial Naive Bayes":
		multiclass_strategy = st.sidebar.selectbox(
			label="Multiclass strategy:",
			options=[
                #'Binary Relevance',
                'Classifier Chain',
                'Label Powerset'])

		# if multiclass_strategy=="Binary Relevance":
		# 	model = joblib.load('models/multinomial_nb_moc.joblib')

		if multiclass_strategy=="Classifier Chain":
			return joblib.load('models/multinomial_nb_cc.joblib')
            
		elif multiclass_strategy=="Label Powerset":
			return joblib.load('models/multinomial_nb_lp.joblib')	

	elif model_selected=="Optimized Naive Bayes":
		return joblib.load('models/grid_search.joblib')        

	elif model_selected=="Random Forest":
		return joblib.load('models/random_forest.joblib')

	elif model_selected=="Support Vector Machine":
		return joblib.load('models/svm.joblib')
    
    