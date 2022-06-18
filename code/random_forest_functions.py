# Random Forest Functions

from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from funcsigs import signature

##############################################################################
# ROC curve
##############################################################################

def roc_curve_custom(classifier, X_test, y_test, y_pred, figure_name, figure_dir):

    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)
    
    roc_auc = auc(fpr, tpr)
    
    plt.figure(1, figsize = (15, 10))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(figure_dir + figure_name)







##############################################################################
# PR curve
##############################################################################


def pr_curve_custom(y_test, y_pred, figure_name, figure_dir):

    average_precision = average_precision_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    
    plt.figure(1, figsize = (15, 10))
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig(figure_dir + figure_name)








