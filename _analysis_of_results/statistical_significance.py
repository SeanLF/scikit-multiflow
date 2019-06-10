import sys
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import Orange
import json
import re

HEIGHT = 2700 * 1.5
WIDTH = 1570 * 1.5
ALPHA = [0.05, 0.01, 0.001]

DASHES = ('-')*23

def safe(string):
    if (string[0] == '[') and string.find('#') < 0 and string.find(';') < 0 and string.find('<<<') < 0:
        return eval(string)
    else: 
        return string

# with open('experiment_results_step3_drift/analysis/all/all.json') as f:
with open('experiment_results_step4_sota/analysis/all/all.json') as f:
# with open('experiment_results_step3_drift/analysis/python.json') as f:
    data = json.load(f)

wilcoxon_results = {}
for key in set([d['params']['key'] for d in data]):
    wilcoxon_results[key] = {'seconds': [], 'kappa_t': []}

nemenyi_results = {}
for key in set([d['params']['key'] for d in data]):
    nemenyi_results[key] = {
        'sig': {'seconds': {'total': 0, 'results': []}, 'kappa_t': {'total': 0, 'results': []}},
        'insig': {'seconds': {'total': 0, 'results': []}, 'kappa_t': {'total': 0, 'results': []}}
    }


for experiment in data:
    results = experiment['results']
    params = experiment['params']

    ascending = False if params['measure'] == 'kappa_t' else True

    print('{}\t=>\t{}/{}'.format(params['measure'], params['key'], params['subkey']))
    to_print = ""

    x = [list(result.values()) for result in results.values()]

    if len(results.keys()) > 2:
        nemenyi_results[params['key']]['sig'][params['measure']]['total'] += 1
        nemenyi_results[params['key']]['insig'][params['measure']]['total'] += 1
        stat, p = friedmanchisquare(*x)
        to_print = "\nFriedman Statistic: {}\tp-value: {}".format(str(stat),str(p))
        # if p < ALPHA[0]:
        # if True:
        groups = np.array([[key]*(len(x[0])) for key in results.keys()]).flatten()
        groups = np.array([re.sub(re.compile(params['subkey']), '', key).replace(' ','') for key in groups])
        groups = [value if value != '' else 'PROBA' for value in groups]
        df = pd.DataFrame({'y': np.concatenate(x),'blocks': np.concatenate([list(range(len(x[0])))]*len(x)),'groups': groups,})
        pn, ranks, qs, qa = sp.posthoc_nemenyi_friedman(df, sort=True, melted=True, y_col='y', block_col='blocks', group_col='groups', ascending=ascending)
        to_print = "{}\n\n\nPost-hoc Nemenyi\n\n\nRanks:\n{}\n{}\n\nCritical difference:\n{}\n{}\n\nDifference:\n{}\n{}\n\nTable:\n{}\n{}".format(to_print, DASHES,str(ranks.sort_values().to_csv()),DASHES,str(qa.to_csv()),DASHES,str(qs.replace([0], '').to_csv()),DASHES,str(pn.to_csv()))
        sig = ''
        if p < ALPHA[0] and len(np.argwhere(np.logical_and(pn.values <= ALPHA[0], pn.values >= 0))) > 0: # check if at least 1 value has p < 0.05
            sig = 'sig'
        else:
            sig = 'insig'

        _diff = [[pn.keys()[x], pn.keys()[y]] for x,y in np.argwhere(np.logical_and(pn.values <= ALPHA[0], pn.values >= 0))]
        _diff.sort()
        nemenyi_results[params['key']][sig][params['measure']]['results'].append([
            p, # p-value
            ranks.sort_values().to_dict(), # ranks
            _diff
        ])
        if True:

            # create heatmap
            fig = plt.figure(figsize=(10, 10))
            heatmap_args = {'linewidths': 0.25, 'linecolor': '0.5', 'clip_on': False, 'square': True, 'cbar_ax_bbox': [0.80, 0.35, 0.04, 0.3]}
            heatmap = sp.sign_plot(pn, **heatmap_args)
            DPI = (fig.get_dpi()) * 2
            fig.set_size_inches(float(HEIGHT)/float(DPI),float(WIDTH)/float(DPI))
            plt.savefig(fname='{}{}_heatmap'.format(params['dir'], params['measure']))
            plt.close(None)

            # add st. sig. ≠ to nemenyi results
            

            qa = (qa / ((2)**.5)) * (len(x) * (len(x) + 1.) / (6.0 * len(x[0]))) ** 0.5
            for alpha in ALPHA:
                # don't output graph if no need
                if len(np.argwhere(np.logical_and(pn.values <= alpha, pn.values >= 0))) > 0:
                # if True:
                    # create nemenyi graph
                    cd = qa.at[1-alpha, 'q_a']
                    filename='{}{}_cd{}.png'.format(params['dir'], params['measure'], str(1.-alpha))
                    Orange.evaluation.graph_ranks(np.array(ranks), np.array(ranks.index), cd=cd, width=15, textspace=2, filename=filename)
                    plt.title('Nemenyi p-values matrix ({}-{}-{})'.format(params['key'], params['subkey'], params['measure']))
                    plt.close(None)

    elif len(results.keys()) == 2:
        try:
            x, y = [np.concatenate(subarray).ravel() for subarray in x]
            (stat, p), r = wilcoxon(x=x, y=y)
            winner = [r.index(max(r)), r.index(min(r))][['kappa_t', 'seconds'].index(params['measure'])]
            being_compared = [re.sub(re.compile(params['subkey']), '', key).replace(' ','') for key in results.keys()]
            being_compared_r = '{} {} {}'.format(being_compared[0], ['>', '<'][winner], being_compared[1])
            if str(p) == 'nan':
                import pdb; pdb.set_trace()

            to_print = "{}\n\nWilcoxon Statistic: {}\tp-value: {}".format(being_compared_r, str(stat),str(p))
            wilcoxon_results[params['key']][params['measure']].append([params['subkey'], p, being_compared[0], r[0], being_compared[1], r[1], being_compared[winner]])
        except Exception:
            pass

    f = open('{}{}.txt'.format(params['dir'], params['measure']), 'w')
    f.write(to_print)
    f.close()

with open('experiment_results_step4_sota/analysis/wilcoxon_results.json', 'w') as outfile:
    json.dump(wilcoxon_results, outfile)

with open('experiment_results_step4_sota/analysis/nemenyi_results.json', 'w') as outfile:
    json.dump(nemenyi_results, outfile)