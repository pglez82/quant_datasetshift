from scipy.stats import wilcoxon

def wilcoxon_test(results_best,results_compare):
    pvalue = wilcoxon(x=results_best,y=results_compare).pvalue
    if pvalue<=0.001:
        return {}
    elif pvalue>0.001 and pvalue<0.05:
        return {'dag':'--rwrap'}
    else:
        return {'ddag':'--rwrap'}