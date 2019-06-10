import pdb
import numpy as np
import pandas as pd
import json
import os

with open('experiment_results_step4_sota/analysis/nemenyi_results.json') as f:
  nemenyi_results = json.load(f)

for key in nemenyi_results.keys():
  csv = []
  for sig in ['sig', 'insig']:
    for subkey in nemenyi_results[key][sig].keys():
      directory = 'experiment_results_step4_sota/analysis/nemenyi'
      if not os.path.exists(directory):
        os.makedirs(directory)
      if len(nemenyi_results[key][sig][subkey]['results']) > 0:
        df = pd.DataFrame(nemenyi_results[key][sig][subkey]['results'], columns=['p-value', 'ranks', 'diff'])
        df.to_csv('{}/{}_{}_{}.csv'.format(directory, key, subkey, sig))
        count = len(df['ranks'])
        df['ranks'] = df['ranks'].apply(lambda r: [key for key in r.keys()])
        df['ranks'] = df['ranks'].apply(lambda r: str(r).replace(":", '').replace('{','').replace('}','').replace("'",'').replace("[", '').replace(']', '').replace(',',' /').replace('ws','').replace('gt',''))
        df['diff'] = df['diff'].apply(lambda d: str(d).replace('],[',"|").replace('"','').replace("'",'').replace('[','').replace(']','').replace(',',' /').replace('ws','').replace('gt',''))
        if sig == 'insig':
          df = df.groupby(by=['ranks']).count()
        else:
          df = df.groupby(by=['ranks', 'diff']).count()
        df.rename({'p-value': 'count'}, axis='columns', inplace=True)

        directory = '{}/_aggregate'.format(directory)
        if not os.path.exists(directory):
          os.makedirs(directory)
        fsig = '\\checkmark' if sig == 'sig' else '$\\times$'
        df = df.assign(sig=len(df)*[fsig]).assign(metric=len(df)*[subkey]).reset_index()
        # import pdb; pdb.set_trace()
        df = df[['metric','sig','ranks', 'diff','count']]
        for value in df.values:
          csv.append(np.asarray(value).flatten().tolist())
  df = pd.DataFrame(csv, columns=['Metric','Stat. Sig.','Ranks', 'Stat. Sig. values','\%']).sort_values(['Metric','Stat. Sig.','Ranks'],ascending=[True,False,True])
  df['\%'] = df['\%'].apply(lambda c: '{0:.2f}\%'.format(100*int(c)/nemenyi_results[key][sig][subkey]['total']))
  df['Metric'] = df['Metric'].apply(lambda m: '$\\kappa_t$' if m == 'kappa_t' else 'Execution time')
  for values in df.values:
    if values[1] == '\\checkmark':
      try:
        new_vvals = []
        vvals = values[3].split(' / ')
        for vals in [vvals[i*2:i*2+2] for i in range(int(len(vvals)/2))]:
          idxs = [values[2].split(' / ').index(el) for el in vals]
          new_vvals.append(' / '.join([vals[idx] for idx in np.argsort(idxs)]))
        values[3] = ' | '.join(new_vvals)
      except Exception:
        pass
    else:
      values[3] = ''
  df = df.sort_values(['Metric','Stat. Sig.','Stat. Sig. values','Ranks'],ascending=[True,False,True,True])

  df.to_csv('{}/{}.csv'.format(directory, key), index=False)

  print()