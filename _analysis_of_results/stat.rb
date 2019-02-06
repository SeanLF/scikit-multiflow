require 'csv'
require 'FileUtils'
require 'json'
require 'byebug'

FOLDERS = {
  step1: ["TUMBLING|PROBABILITY|100gt|100ws","TUMBLING|PROBABILITY|100gt|75ws","TUMBLING|PROBABILITY|100gt|50ws","TUMBLING|PROBABILITY|100gt|25ws","TUMBLING|PROBABILITY|100gt|10ws","TUMBLING|PROBABILITY|100gt|5ws","SLIDING|PROBABILITY|100gt|100ws","SLIDING|PROBABILITY|100gt|75ws","SLIDING|PROBABILITY|100gt|50ws","SLIDING|PROBABILITY|100gt|25ws","SLIDING|PROBABILITY|100gt|10ws","SLIDING|PROBABILITY|100gt|5ws","HYBRID|PROBABILITY|100gt|100ws","HYBRID|PROBABILITY|100gt|75ws","HYBRID|PROBABILITY|100gt|50ws","HYBRID|PROBABILITY|100gt|25ws","HYBRID|PROBABILITY|100gt|10ws","HYBRID|PROBABILITY|100gt|5ws","HYBRID|BOOLEAN|100gt|5ws","TUMBLING|BOOLEAN|100gt|5ws","SLIDING|BOOLEAN|100gt|5ws","HYBRID|W_AVG_PROBABILITY|100gt|5ws","TUMBLING|W_AVG_PROBABILITY|100gt|5ws","SLIDING|W_AVG_PROBABILITY|100gt|5ws","HYBRID|AVG_W_PROBABILITY|100gt|5ws","TUMBLING|AVG_W_PROBABILITY|100gt|5ws","SLIDING|AVG_W_PROBABILITY|100gt|5ws","HYBRID|BOOLEAN|100gt|10ws","TUMBLING|BOOLEAN|100gt|10ws","SLIDING|BOOLEAN|100gt|10ws","HYBRID|W_AVG_PROBABILITY|100gt|10ws","TUMBLING|W_AVG_PROBABILITY|100gt|10ws","SLIDING|W_AVG_PROBABILITY|100gt|10ws","HYBRID|AVG_W_PROBABILITY|100gt|10ws","TUMBLING|AVG_W_PROBABILITY|100gt|10ws","SLIDING|AVG_W_PROBABILITY|100gt|10ws","HYBRID|BOOLEAN|100gt|25ws","TUMBLING|BOOLEAN|100gt|25ws","SLIDING|BOOLEAN|100gt|25ws","HYBRID|W_AVG_PROBABILITY|100gt|25ws","TUMBLING|W_AVG_PROBABILITY|100gt|25ws","SLIDING|W_AVG_PROBABILITY|100gt|25ws","HYBRID|AVG_W_PROBABILITY|100gt|25ws","TUMBLING|AVG_W_PROBABILITY|100gt|25ws","SLIDING|AVG_W_PROBABILITY|100gt|25ws","HYBRID|BOOLEAN|100gt|50ws","TUMBLING|BOOLEAN|100gt|50ws","SLIDING|BOOLEAN|100gt|50ws","HYBRID|W_AVG_PROBABILITY|100gt|50ws","TUMBLING|W_AVG_PROBABILITY|100gt|50ws","SLIDING|W_AVG_PROBABILITY|100gt|50ws","HYBRID|AVG_W_PROBABILITY|100gt|50ws","TUMBLING|AVG_W_PROBABILITY|100gt|50ws","SLIDING|AVG_W_PROBABILITY|100gt|50ws","HYBRID|BOOLEAN|100gt|75ws","TUMBLING|BOOLEAN|100gt|75ws","SLIDING|BOOLEAN|100gt|75ws","HYBRID|W_AVG_PROBABILITY|100gt|75ws","TUMBLING|W_AVG_PROBABILITY|100gt|75ws","SLIDING|W_AVG_PROBABILITY|100gt|75ws","HYBRID|AVG_W_PROBABILITY|100gt|75ws","TUMBLING|AVG_W_PROBABILITY|100gt|75ws","SLIDING|AVG_W_PROBABILITY|100gt|75ws","HYBRID|BOOLEAN|100gt|100ws","TUMBLING|BOOLEAN|100gt|100ws","SLIDING|BOOLEAN|100gt|100ws","HYBRID|W_AVG_PROBABILITY|100gt|100ws","TUMBLING|W_AVG_PROBABILITY|100gt|100ws","SLIDING|W_AVG_PROBABILITY|100gt|100ws","HYBRID|AVG_W_PROBABILITY|100gt|100ws","TUMBLING|AVG_W_PROBABILITY|100gt|100ws","SLIDING|AVG_W_PROBABILITY|100gt|100ws"],
  step2: ["HYBRID|AVG_W_PROBABILITY|60gt|25ws","HYBRID|AVG_W_PROBABILITY|60gt|50ws","HYBRID|AVG_W_PROBABILITY|60gt|75ws","HYBRID|AVG_W_PROBABILITY|60gt|100ws","HYBRID|AVG_W_PROBABILITY|70gt|25ws","HYBRID|AVG_W_PROBABILITY|70gt|50ws","HYBRID|AVG_W_PROBABILITY|70gt|75ws","HYBRID|AVG_W_PROBABILITY|70gt|100ws","HYBRID|AVG_W_PROBABILITY|80gt|25ws","HYBRID|AVG_W_PROBABILITY|80gt|50ws","HYBRID|AVG_W_PROBABILITY|80gt|75ws","HYBRID|AVG_W_PROBABILITY|80gt|100ws","HYBRID|AVG_W_PROBABILITY|90gt|25ws","HYBRID|AVG_W_PROBABILITY|90gt|50ws","HYBRID|AVG_W_PROBABILITY|90gt|75ws","HYBRID|AVG_W_PROBABILITY|90gt|100ws","HYBRID|AVG_W_PROBABILITY|100gt|25ws","HYBRID|AVG_W_PROBABILITY|100gt|50ws","HYBRID|AVG_W_PROBABILITY|100gt|75ws","HYBRID|AVG_W_PROBABILITY|100gt|100ws","HYBRID|PROBABILITY|60gt|25ws","HYBRID|PROBABILITY|60gt|50ws","HYBRID|PROBABILITY|60gt|75ws","HYBRID|PROBABILITY|60gt|100ws","HYBRID|PROBABILITY|70gt|25ws","HYBRID|PROBABILITY|70gt|50ws","HYBRID|PROBABILITY|70gt|75ws","HYBRID|PROBABILITY|70gt|100ws","HYBRID|PROBABILITY|80gt|25ws","HYBRID|PROBABILITY|80gt|50ws","HYBRID|PROBABILITY|80gt|75ws","HYBRID|PROBABILITY|80gt|100ws","HYBRID|PROBABILITY|90gt|25ws","HYBRID|PROBABILITY|90gt|50ws","HYBRID|PROBABILITY|90gt|75ws","HYBRID|PROBABILITY|90gt|100ws","HYBRID|PROBABILITY|100gt|25ws","HYBRID|PROBABILITY|100gt|50ws","HYBRID|PROBABILITY|100gt|75ws","HYBRID|PROBABILITY|100gt|100ws","HYBRID|W_AVG_PROBABILITY|60gt|25ws","HYBRID|W_AVG_PROBABILITY|60gt|50ws","HYBRID|W_AVG_PROBABILITY|60gt|75ws","HYBRID|W_AVG_PROBABILITY|60gt|100ws","HYBRID|W_AVG_PROBABILITY|70gt|25ws","HYBRID|W_AVG_PROBABILITY|70gt|50ws","HYBRID|W_AVG_PROBABILITY|70gt|75ws","HYBRID|W_AVG_PROBABILITY|70gt|100ws","HYBRID|W_AVG_PROBABILITY|80gt|25ws","HYBRID|W_AVG_PROBABILITY|80gt|50ws","HYBRID|W_AVG_PROBABILITY|80gt|75ws","HYBRID|W_AVG_PROBABILITY|80gt|100ws","HYBRID|W_AVG_PROBABILITY|90gt|25ws","HYBRID|W_AVG_PROBABILITY|90gt|50ws","HYBRID|W_AVG_PROBABILITY|90gt|75ws","HYBRID|W_AVG_PROBABILITY|90gt|100ws","HYBRID|W_AVG_PROBABILITY|100gt|25ws","HYBRID|W_AVG_PROBABILITY|100gt|50ws","HYBRID|W_AVG_PROBABILITY|100gt|75ws","HYBRID|W_AVG_PROBABILITY|100gt|100ws","SLIDING|AVG_W_PROBABILITY|60gt|25ws","SLIDING|AVG_W_PROBABILITY|60gt|50ws","SLIDING|AVG_W_PROBABILITY|60gt|75ws","SLIDING|AVG_W_PROBABILITY|60gt|100ws","SLIDING|AVG_W_PROBABILITY|70gt|25ws","SLIDING|AVG_W_PROBABILITY|70gt|50ws","SLIDING|AVG_W_PROBABILITY|70gt|75ws","SLIDING|AVG_W_PROBABILITY|70gt|100ws","SLIDING|AVG_W_PROBABILITY|80gt|25ws","SLIDING|AVG_W_PROBABILITY|80gt|50ws","SLIDING|AVG_W_PROBABILITY|80gt|75ws","SLIDING|AVG_W_PROBABILITY|80gt|100ws","SLIDING|AVG_W_PROBABILITY|90gt|25ws","SLIDING|AVG_W_PROBABILITY|90gt|50ws","SLIDING|AVG_W_PROBABILITY|90gt|75ws","SLIDING|AVG_W_PROBABILITY|90gt|100ws","SLIDING|AVG_W_PROBABILITY|100gt|25ws","SLIDING|AVG_W_PROBABILITY|100gt|50ws","SLIDING|AVG_W_PROBABILITY|100gt|75ws","SLIDING|AVG_W_PROBABILITY|100gt|100ws","SLIDING|PROBABILITY|60gt|25ws","SLIDING|PROBABILITY|60gt|50ws","SLIDING|PROBABILITY|60gt|75ws","SLIDING|PROBABILITY|60gt|100ws","SLIDING|PROBABILITY|70gt|25ws","SLIDING|PROBABILITY|70gt|50ws","SLIDING|PROBABILITY|70gt|75ws","SLIDING|PROBABILITY|70gt|100ws","SLIDING|PROBABILITY|80gt|25ws","SLIDING|PROBABILITY|80gt|50ws","SLIDING|PROBABILITY|80gt|75ws","SLIDING|PROBABILITY|80gt|100ws","SLIDING|PROBABILITY|90gt|25ws","SLIDING|PROBABILITY|90gt|50ws","SLIDING|PROBABILITY|90gt|75ws","SLIDING|PROBABILITY|90gt|100ws","SLIDING|PROBABILITY|100gt|25ws","SLIDING|PROBABILITY|100gt|50ws","SLIDING|PROBABILITY|100gt|75ws","SLIDING|PROBABILITY|100gt|100ws","SLIDING|W_AVG_PROBABILITY|60gt|25ws","SLIDING|W_AVG_PROBABILITY|60gt|50ws","SLIDING|W_AVG_PROBABILITY|60gt|75ws","SLIDING|W_AVG_PROBABILITY|60gt|100ws","SLIDING|W_AVG_PROBABILITY|70gt|25ws","SLIDING|W_AVG_PROBABILITY|70gt|50ws","SLIDING|W_AVG_PROBABILITY|70gt|75ws","SLIDING|W_AVG_PROBABILITY|70gt|100ws","SLIDING|W_AVG_PROBABILITY|80gt|25ws","SLIDING|W_AVG_PROBABILITY|80gt|50ws","SLIDING|W_AVG_PROBABILITY|80gt|75ws","SLIDING|W_AVG_PROBABILITY|80gt|100ws","SLIDING|W_AVG_PROBABILITY|90gt|25ws","SLIDING|W_AVG_PROBABILITY|90gt|50ws","SLIDING|W_AVG_PROBABILITY|90gt|75ws","SLIDING|W_AVG_PROBABILITY|90gt|100ws","SLIDING|W_AVG_PROBABILITY|100gt|25ws","SLIDING|W_AVG_PROBABILITY|100gt|50ws","SLIDING|W_AVG_PROBABILITY|100gt|75ws","SLIDING|W_AVG_PROBABILITY|100gt|100ws","TUMBLING|AVG_W_PROBABILITY|60gt|25ws","TUMBLING|AVG_W_PROBABILITY|60gt|50ws","TUMBLING|AVG_W_PROBABILITY|60gt|75ws","TUMBLING|AVG_W_PROBABILITY|60gt|100ws","TUMBLING|AVG_W_PROBABILITY|70gt|25ws","TUMBLING|AVG_W_PROBABILITY|70gt|50ws","TUMBLING|AVG_W_PROBABILITY|70gt|75ws","TUMBLING|AVG_W_PROBABILITY|70gt|100ws","TUMBLING|AVG_W_PROBABILITY|80gt|25ws","TUMBLING|AVG_W_PROBABILITY|80gt|50ws","TUMBLING|AVG_W_PROBABILITY|80gt|75ws","TUMBLING|AVG_W_PROBABILITY|80gt|100ws","TUMBLING|AVG_W_PROBABILITY|90gt|25ws","TUMBLING|AVG_W_PROBABILITY|90gt|50ws","TUMBLING|AVG_W_PROBABILITY|90gt|75ws","TUMBLING|AVG_W_PROBABILITY|90gt|100ws","TUMBLING|AVG_W_PROBABILITY|100gt|25ws","TUMBLING|AVG_W_PROBABILITY|100gt|50ws","TUMBLING|AVG_W_PROBABILITY|100gt|75ws","TUMBLING|AVG_W_PROBABILITY|100gt|100ws","TUMBLING|PROBABILITY|60gt|25ws","TUMBLING|PROBABILITY|60gt|50ws","TUMBLING|PROBABILITY|60gt|75ws","TUMBLING|PROBABILITY|60gt|100ws","TUMBLING|PROBABILITY|70gt|25ws","TUMBLING|PROBABILITY|70gt|50ws","TUMBLING|PROBABILITY|70gt|75ws","TUMBLING|PROBABILITY|70gt|100ws","TUMBLING|PROBABILITY|80gt|25ws","TUMBLING|PROBABILITY|80gt|50ws","TUMBLING|PROBABILITY|80gt|75ws","TUMBLING|PROBABILITY|80gt|100ws","TUMBLING|PROBABILITY|90gt|25ws","TUMBLING|PROBABILITY|90gt|50ws","TUMBLING|PROBABILITY|90gt|75ws","TUMBLING|PROBABILITY|90gt|100ws","TUMBLING|PROBABILITY|100gt|25ws","TUMBLING|PROBABILITY|100gt|50ws","TUMBLING|PROBABILITY|100gt|75ws","TUMBLING|PROBABILITY|100gt|100ws","TUMBLING|W_AVG_PROBABILITY|60gt|25ws","TUMBLING|W_AVG_PROBABILITY|60gt|50ws","TUMBLING|W_AVG_PROBABILITY|60gt|75ws","TUMBLING|W_AVG_PROBABILITY|60gt|100ws","TUMBLING|W_AVG_PROBABILITY|70gt|25ws","TUMBLING|W_AVG_PROBABILITY|70gt|50ws","TUMBLING|W_AVG_PROBABILITY|70gt|75ws","TUMBLING|W_AVG_PROBABILITY|70gt|100ws","TUMBLING|W_AVG_PROBABILITY|80gt|25ws","TUMBLING|W_AVG_PROBABILITY|80gt|50ws","TUMBLING|W_AVG_PROBABILITY|80gt|75ws","TUMBLING|W_AVG_PROBABILITY|80gt|100ws","TUMBLING|W_AVG_PROBABILITY|90gt|25ws","TUMBLING|W_AVG_PROBABILITY|90gt|50ws","TUMBLING|W_AVG_PROBABILITY|90gt|75ws","TUMBLING|W_AVG_PROBABILITY|90gt|100ws","TUMBLING|W_AVG_PROBABILITY|100gt|25ws","TUMBLING|W_AVG_PROBABILITY|100gt|50ws","TUMBLING|W_AVG_PROBABILITY|100gt|75ws","TUMBLING|W_AVG_PROBABILITY|100gt|100ws"],
  top10: ["SLIDING|PROBABILITY|100gt|50ws","SLIDING|W_AVG_PROBABILITY|100gt|75ws","SLIDING|PROBABILITY|90gt|50ws","SLIDING|PROBABILITY|100gt|25ws","SLIDING|PROBABILITY|90gt|75ws","SLIDING|PROBABILITY|90gt|25ws","SLIDING|W_AVG_PROBABILITY|100gt|50ws","SLIDING|W_AVG_PROBABILITY|100gt|100ws","SLIDING|W_AVG_PROBABILITY|90gt|25ws","SLIDING|W_AVG_PROBABILITY|100gt|25ws","HYBRID|PROBABILITY|90gt|100ws","HYBRID|PROBABILITY|70gt|100ws","HYBRID|PROBABILITY|80gt|100ws","HYBRID|PROBABILITY|60gt|100ws","HYBRID|PROBABILITY|100gt|100ws","HYBRID|W_AVG_PROBABILITY|70gt|100ws","HYBRID|W_AVG_PROBABILITY|60gt|100ws","HYBRID|W_AVG_PROBABILITY|100gt|100ws","HYBRID|W_AVG_PROBABILITY|90gt|100ws","HYBRID|W_AVG_PROBABILITY|80gt|100ws","SLIDING|PROBABILITY|100gt|100ws","SLIDING|PROBABILITY|90gt|100ws","HYBRID|PROBABILITY|100gt|75ws","TUMBLING|PROBABILITY|100gt|100ws","SLIDING|W_AVG_PROBABILITY|90gt|100ws","SLIDING|W_AVG_PROBABILITY|80gt|100ws"]
}
def gsub_m(folder)
  subs=[['TUMBLING','t'],['HYBRID','h'],['SLIDING','s'],["BOOLEAN",'bool'],["AVG_W_PROBABILITY",'avgw'],["W_AVG_PROBABILITY",'wavg'],["PROBABILITY",'prob']]

  for sub in subs
    folder = folder.gsub(sub[0], sub[1])
  end
  return folder
end

def fdo(kappa_t_OR_seconds, folders_key)
  measure_for_regex = kappa_t_OR_seconds ? "ClassificationMeasurements:" : "Evaluation time:"
  measure = kappa_t_OR_seconds ? 'kappa_t' : 'seconds'
  kappa_or_seconds_regex = kappa_t_OR_seconds ? /:\s(-?\d\.\d+)/ : /(\d+\.\d+)/
  results = {}
  top_dir = './experiment_results_step2_gt'

  for folder in FOLDERS[folders_key]
    folder_files = []
    folder_files = %x[egrep #{measure_for_regex} #{top_dir}/#{folder.gsub('|', '\|')}/*.txt].split("\n")
    if (folder_files.count != 18)
      byebug
    end
    
    g = folder_files.map{|f| f.split(".txt")}.group_by{|f| f[0][/VOTING_ENSEMBLE\[([a-z]+(_noise_0\.\d)?).*\]/i, 1]}
    groups = {}
    if kappa_t_OR_seconds
      g.keys.each{|k| groups[k] = g[k].map{|v| v[1].scan(kappa_or_seconds_regex).flatten[2]}.map &:to_f}
    else
      g.keys.each{|k| groups[k] = g[k].map{|v| v[1].scan(kappa_or_seconds_regex)}.flatten.map &:to_f}
    end
    groups.keys.each {|k| groups[k] = groups[k].sum / groups[k].count}
    results[gsub_m(folder)] = groups
  end

  data, header = [], results[results.keys.first].keys.prepend('params \ dataset')
  results.keys.each {|key| data << results[key].values.prepend(key) }
  top_analysis_folder = top_dir + '/analysis'

  dir = "#{top_analysis_folder}/top10s/"
  FileUtils.mkdir_p(dir)
  csv_path = dir + measure + '.csv'
  CSV.open(csv_path, "wb") do |csv|
    csv << header
    for row in data
      csv << row
    end
  end

  # run Python program to evaluate statistical significance using numpy and external libraries
  params = {key: '', subkey: '', measure: measure, dir: dir}
  system "python3 statistical_significance.py '#{JSON.fast_generate(results)}' '#{JSON.fast_generate(params)}'"
end

fdo(true, :top10)
fdo(false, :top10)