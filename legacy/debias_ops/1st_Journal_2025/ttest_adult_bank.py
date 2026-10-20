from scipy.stats import ttest_rel
import numpy as np

adult_accuracy_raw = np.array([0.8102454390943489, 0.004389357388134816, 0.8029481011362473, 0.8153153153153153])
adult_accuracy_with_cleaning = np.array([0.8099383242668123, 0.004695428035739532, 0.8034599242501791, 0.8171580671580672])

t_stat, p_val = ttest_rel(adult_accuracy_with_cleaning, adult_accuracy_raw)
print("Accuracy (adult dataset) paired t-test:", t_stat, p_val)

bank_accuracy_raw = np.array([0.5892755282626546, 0.22434648396065296, 0.2099093120990931, 0.8533672453831693])
bank_accuracy_with_cleaning = np.array([0.6378714878504609, 0.1316800189473956, 0.4810882548108825, 0.8300342806590734])

t_stat, p_val = ttest_rel(bank_accuracy_with_cleaning, bank_accuracy_raw)
print("Accuracy (bank dataset) paired t-test:", t_stat, p_val)
    
adult_precision_raw = np.array([0.816628085542054, 0.004080824005535164, 0.808657381587068, 0.8196922619859043])
adult_precision_with_cleaning = np.array([0.8166993065417003, 0.004067805846247965, 0.8093819633802706, 0.821112377108836])

t_stat, p_val = ttest_rel(adult_precision_with_cleaning, adult_precision_raw)
print("Precision (adult dataset) paired t-test:", t_stat, p_val)

bank_precision_raw = np.array([0.8217724741885373, 0.032901171136882706, 0.7875247071917296, 0.878019291318313])
bank_precision_with_cleaning = np.array([0.8352476003720906, 0.028917129440954176, 0.7938713383748424, 0.8720192868522794])

t_stat, p_val = ttest_rel(bank_precision_with_cleaning, bank_precision_raw)
print("Precision (bank dataset) paired t-test:", t_stat, p_val)

adult_recall_raw = np.array([0.8102454390943489, 0.004389357388134816, 0.8029481011362473, 0.8153153153153153])
adult_recall_with_cleaning = np.array([0.8099383242668123, 0.004695428035739532, 0.8034599242501791, 0.8171580671580672])

t_stat, p_val = ttest_rel(adult_recall_with_cleaning, adult_recall_raw)
print("Recall (adult dataset) paired t-test:", t_stat, p_val)

bank_recall_raw = np.array([0.5892755282626546, 0.22434648396065296, 0.2099093120990931, 0.8533672453831693])
bank_recall_with_cleaning = np.array([0.6378714878504609, 0.1316800189473956, 0.4810882548108825, 0.8300342806590734])

t_stat, p_val = ttest_rel(bank_recall_with_cleaning, bank_recall_raw)
print("Recall (bank dataset) paired t-test:", t_stat, p_val)

adult_f1_raw = np.array([0.8129912129744573, 0.004174875568220266, 0.8054812889793582, 0.8171472313937872])
adult_f1_with_cleaning = np.array([0.8128275570571539, 0.004365446392235888, 0.8060748564082005, 0.818947382732934])

t_stat, p_val = ttest_rel(adult_f1_with_cleaning, adult_f1_raw)
print("F1 (adult dataset) paired t-test:", t_stat, p_val)

bank_f1_raw = np.array([0.6283136233527733, 0.22841594869347037, 0.21309984763110967, 0.8635292981995274])
bank_f1_with_cleaning = np.array([0.6939666923322033, 0.10734910225055731, 0.567986643017243, 0.8464876618451349])

t_stat, p_val = ttest_rel(bank_f1_with_cleaning, bank_f1_raw)
print("F1 (bank dataset) paired t-test:", t_stat, p_val)

adult_roc_auc_raw = np.array([0.7551244338221241, 0.006004511813990643, 0.7432621636017428, 0.7592048730268858])
adult_roc_auc_with_cleaning = np.array([0.7554551077174306, 0.005643361268279965, 0.744478169792257, 0.759817362957005])

t_stat, p_val = ttest_rel(adult_roc_auc_with_cleaning, adult_roc_auc_raw)
print("ROC AUC (adult dataset) paired t-test:", t_stat, p_val)

bank_roc_auc_raw = np.array([0.5819089690328273, 0.08873477081436569, 0.4863018806990162, 0.7312518273274677])
bank_roc_auc_with_cleaning = np.array([0.6092634484744195, 0.0736827757285356, 0.5011800538695539, 0.7229592229286244])

t_stat, p_val = ttest_rel(bank_roc_auc_with_cleaning, bank_roc_auc_raw)
print("ROC AUC (bank dataset) paired t-test:", t_stat, p_val)

adult_balanced_accuracy_raw = np.array([0.7549866247645052, 0.005998967571967712, 0.7431037055020819, 0.7590444867229611])
adult_balanced_accuracy_with_cleaning = np.array([0.7554007190894613, 0.005694806426772184, 0.7443195677967556, 0.7598173629570051])

t_stat, p_val = ttest_rel(adult_balanced_accuracy_with_cleaning, adult_balanced_accuracy_raw)
print("Balanced Accuracy (adult dataset) paired t-test:", t_stat, p_val)

bank_balanced_accuracy_raw = np.array([0.5819089690328273, 0.08873477081436569, 0.4863018806990162, 0.7312518273274677])
bank_balanced_accuracy_with_cleaning = np.array([0.6092634484744195, 0.07368277572853565, 0.5011800538695539, 0.722959222928624])

t_stat, p_val = ttest_rel(bank_balanced_accuracy_with_cleaning, bank_balanced_accuracy_raw)
print("Balanced Accuracy (bank dataset) paired t-test:", t_stat, p_val)