import sys
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import Orange
import json

def safe(string):
    if (string[0] == '[') and string.find('#') < 0 and string.find(';') < 0 and string.find('<<<') < 0:
        return eval(string)
    else: 
        return string

args = sys.argv
results = json.loads(args[1])
params = json.loads(args[2])

HEIGHT = 2700 * 1.5
WIDTH = 1570 * 1.5
ALPHA = [0.05, 0.01, 0.001]

DASHES = ('-')*23

ascending = False if params['measure'] == 'kappa_t' else True

print('{}\t=>\t{}/{}'.format(params['measure'], params['key'], params['subkey']))

x = [list(result.values()) for result in results.values()]
stat, p = friedmanchisquare(*x)
to_print = "\nFriedman Statistic: {}\tp-value: {}".format(str(stat),str(p))
# if p < ALPHA[0]:
if True:
    groups = np.array([[key]*(len(x[0])) for key in results.keys()]).flatten()
    df = pd.DataFrame({'y': np.concatenate(x),'blocks': np.concatenate([list(range(len(x[0])))]*len(x)),'groups': groups,})
    pn, ranks, qs, qa = sp.posthoc_nemenyi_friedman(df, sort=True, melted=True, y_col='y', block_col='blocks', group_col='groups', ascending=ascending)
    to_print = "{}\n\n\nPost-hoc Nemenyi\n\n\nRanks:\n{}\n{}\n\nCritical difference:\n{}\n{}\n\nDifference:\n{}\n{}\n\nTable:\n{}\n{}".format(to_print, DASHES,str(ranks.sort_values().to_csv()),DASHES,str(qa.to_csv()),DASHES,str(qs.replace([0], '').to_csv()),DASHES,str(pn.to_csv()))
    if len(np.argwhere(np.logical_and(pn.values <= ALPHA[0], pn.values >= 0))) > 0: # check if at least 1 value has p < 0.05
        fig = plt.figure(figsize=(10, 10))
        heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
        heatmap = sp.sign_plot(pn, **heatmap_args)
        DPI = (fig.get_dpi()) * 2
        fig.set_size_inches(float(HEIGHT)/float(DPI),float(WIDTH)/float(DPI))
        plt.savefig(fname='{}{}_heatmap'.format(params['dir'], params['measure']))
        plt.close(None)

        names = np.array(ranks.index)
        avranks =  np.array(ranks)
        qa = (qa / ((2)**.5)) * (len(x) * (len(x) + 1.) / (6.0 * len(x[0]))) ** 0.5
        for alpha in ALPHA:
            # don't output graph if no need
            if len(np.argwhere(np.logical_and(pn.values <= alpha, pn.values >= 0))) > 0:
            # if True:
                cd = qa.at[1-alpha, 'q_a']
                filename='{}{}_cd{}.png'.format(params['dir'], params['measure'], str(1.-alpha))
                plt.title('Nemenyi p-values matrix ({}-{}-{})'.format(params['key'], params['subkey'], params['measure']))
                Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=15, textspace=2, filename=filename)
                plt.close(None)

f = open('{}{}.txt'.format(params['dir'], params['measure']), 'w')
f.write(to_print)
f.close()