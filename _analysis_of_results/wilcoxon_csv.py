import numpy as np
import pandas as pd
import json

with open('experiment_results_step3_drift/analysis/wilcoxon_results.json') as f:
    global_results = json.load(f)

mega_output = []
for key in global_results.keys():
    for subkey in global_results[key].keys():
        if len(global_results[key][subkey]) < 1:
            next
        else:
            alphas = {
                'insig': { 'min': 0.05,  'max': 100,  'count': 0, 'csv': [], 'df': None, 'file': 'insignificant' },
                'sig':   { 'min': 0.01,  'max': 0.05, 'count': 0, 'csv': [], 'df': None, 'file': 'significant'   },
                'vsig':  { 'min': 0.001, 'max': 0.01, 'count': 0,                                                },
                'usig':  { 'min': 0,     'max': 0.01, 'count': 0,                                                },
            }
            alpha_keys = [key for key in alphas.keys()]

            # sig5, sig1, sig01, insig = [], [], [], []
            for result in global_results[key][subkey]:
                pval = result[6]
                for alpha in alphas.keys():
                    if alphas[alpha]['min'] < pval and pval <= alphas[alpha]['max']:
                        alphas[alpha]['count'] += 1
                        if alpha == 'insig':
                            alphas[alpha]['csv'].append(result)
                        else:
                            alphas['sig']['csv'].append(result)
            base_file = 'experiment_results_step3_drift/analysis/wilcoxon_{}_{}'.format(key, subkey)
            
            for alpha in ['insig', 'sig']:
                alphas[alpha]['file'] = '{}_{}.csv'.format(base_file, alphas[alpha]['file'])
                alphas[alpha]['df'] = pd.DataFrame(np.array(alphas[alpha]['csv']))
                alphas[alpha]['df'].to_csv(alphas[alpha]['file'], header=['0','1','2','3','4','5','p-value', 'param1', 'count', 'param2', 'count', 'winner'])
            lens = {}
            lens['sig'] = np.sum([alphas['sig']['count'], alphas['vsig']['count'], alphas['usig']['count']])
            lens['total'] = lens['sig'] + alphas['insig']['count']
            counts = [['{}%'.format(int(100*count/np.sum(counts))) for count in counts] for counts in [np.array(alphas[alpha]['df'].sort_values(by=11).groupby([11]).count()[0]) for alpha in ['sig','insig']]]
            dicts = []
            for i, val in enumerate(['sig', 'insig']):
                dicts.append(alphas[val]['df'].sort_values(by=11).groupby([11]).count()[i].to_dict())
            dicts = dict(zip(dicts[0].keys(), [[list(dicts[0].values())[j], counts[0][j], list(dicts[0+1].values())[j], counts[0+1][j]] for j in range(2)]))

            mega_output.append([
                key,
                subkey,
                '{}%'.format(int(100*float(alphas['sig']['count'])/float(lens['total']))),
                '{}%'.format(int(100*float(alphas['vsig']['count'])/float(lens['total']))),
                '{}%'.format(int(100*float(alphas['usig']['count'])/float(lens['total']))),
                '{}%'.format(int(100*float(lens['sig'])/float(lens['total']))),
                '{}'.format(dicts)
            ])
            # import pdb; pdb.set_trace()
df=pd.DataFrame(mega_output, columns=['param', 'measure', '0.05', '0.01', '0.001', 'significant %', 'breakdown'])
df.to_csv('experiment_results_step3_drift/analysis/wilcoxon.csv')
