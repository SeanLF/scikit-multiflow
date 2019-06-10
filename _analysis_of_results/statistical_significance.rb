###############################################################################
#                          Author: Sean Floyd                                 #
###############################################################################
# The functions in this file generate combinations of parameters, parses the  #
# respective files in an experiment results directory, collects various       #
# metrics, then outputs a CSV file per metric and experiment, then computes   #
# statistical significance tests using Python (statistical_significance.py)   #
# and outputs files showing the results of post-hoc Nemenyi tests if the      #
# Friedman test indicated an initial statistical significance.                #
###############################################################################

require 'csv'
require 'FileUtils'
require 'json'
require 'byebug'

def _merge_array_of_hashes(array)
  """
  Merges multiple hashes in an array into a single hash

  --Example
  >>> array = [{key: value}, {otherKey: otherValue}, {yetAnother: ugh}]
  >>> _merge_array_of_hashes(array)
  [{key: value, otherKey: otherValue, yetAnother: ugh}]
  """
  hash = {}
  array.each {|el| hash = hash.merge(el)}
  return hash
end

def _combination_arrays(array)
  """ 
  Creates all combinations of all subarrays in array.

  --Example
  >>> array = [[1], [2, 3], [4, 5, 6]]
  >>>> _combination_arrays(array)
  [[[1, 2], 4], [[1, 2], 5], [[1, 2], 6], [[1, 3], 4], [[1, 3], 5], [[1, 3], 6]]

  --Note--
  By flattening each subarray, we can obtain more readable output combinations.
  >>> _combination_arrays(array).map(&:flatten)
  [[1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 3, 4], [1, 3, 5], [1, 3, 6]]
  """

  count = array.count
  a = array[0].product(array[1])
  array[2, count-1].each {|el| a = a.product(el)} if count > 2
  return a
end

def generate_combinations(params, reject_key, &format)
  comparing, combinations = params.keys.reject{|k| k==reject_key}, {}
  comparing.each_with_index do |to_compare, index|
    # generate combinations
    other_keys = (comparing[0,index] + comparing[index+1, comparing.count-1])
    others = other_keys.map{|k| params[k].map{|v| {k => v}}}
    combination = _combination_arrays(others)

    # add to combinations hash
    combinations[to_compare] = {}
    
    for c in params[to_compare]
      combi = combination.product([{to_compare => c}]).map(&:flatten).map{|a| _merge_array_of_hashes(a)}
      x=combination.map(&:flatten).each do |c|
        h=_merge_array_of_hashes(c)
        
        # experiment key
        a = []
        if (h.keys.include?(:window_type) and :window_type != reject_key)
          a << h[:window_type]
        end
        if (h.keys.include?(:voting_type) and :voting_type != reject_key)
          a << h[:voting_type]
        end
        if (h.keys.include?(:ground_truth) and :ground_truth != reject_key)
          a << "#{h[:ground_truth]}gt"
        end
        if (h.keys.include?(:batch_size) and :batch_size != reject_key)
          a << "#{h[:batch_size]}ws"
        end
        if (h.keys.include?(:drift_reset_type) and :drift_reset_type != reject_key)
          a << "#{h[:drift_reset_type]}"
        end
        if (h.keys.include?(:drift_detector_count) and :drift_detector_count != reject_key)
          a << "#{h[:drift_detector_count]}"
        end
        if (h.keys.include?(:drift_content) and :drift_content != reject_key)
          a << "#{h[:drift_content]}"
        end

        a = a.join('|')
        
        # to_compare_together
        cbs = params[to_compare].map{|tc| {to_compare => tc}}.product([h]).map{|a| _merge_array_of_hashes(a)}
        
        cbs.map! {|cb| format.call(cb)}
        combinations[to_compare][a] = cbs
      end
    end
  end
  return combinations
end

def compute_statistical_significance(params, top_dir, reject_key)
  top_analysis_folder = top_dir + '/analysis'
    
  combinations = generate_combinations(params, reject_key) {|c| "#{c[:window_type]}|#{c[:voting_type]}|#{c[:ground_truth]}gt|#{c[:batch_size]}ws|#{c[:drift_reset_type]}|#{c[:drift_detector_count]}|#{c[:drift_content]}"}
  to_be_JSONed = []
  for key in combinations.keys
    for subkey in combinations[key].keys.reject{|k| k[/W\_AVG\_PROBABILITY\|.*\|.*\|ONE\_PER\_CLASSIFIER\|WEIGHTED\_PROBABILITY/]}.reject{|k| k[/(^|\|)PROBABILITY\|.*\|.*\|.*\|WEIGHTED\_PROBABILITY/]}
      for kappa_t_OR_seconds in [true, false]
        experiment_name = "#{key.to_s} - #{subkey}" # example: batch_size - SLIDING|HARD means we're comparing all batch sizes over SLIDING windows and HARD voting and 100 ground truth
        measure_for_regex = kappa_t_OR_seconds ? "ClassificationMeasurements:" : "Evaluation time:"
        measure = kappa_t_OR_seconds ? 'kappa_t' : 'seconds'
        kappa_or_seconds_regex = kappa_t_OR_seconds ? /:\s(-?\d\.\d+)/ : /(\d+\.\d+)/
        
        files_to_search = combinations[key][subkey].map{|c| c.gsub('||','|*|')} #TODO: check
        results = {} # now grouped by 
        for file_to_search in files_to_search
          # search For results files
          file_to_search=file_to_search.gsub("|gt|","|#{params[:ground_truth].first}gt|")
          files_found = %x[egrep "#{measure_for_regex}" #{top_dir}/*/#{file_to_search.gsub('|', '\|')}/*.txt].split("\n")
          if files_found.empty?
            break
          end

          # group by dataset
          g = files_found.map{|f| f.split(".txt")}.group_by{|f| f[0][/VOTING_ENSEMBLE\[([a-z]+(_noise_0\.\d)?).*\]/i, 1]}
          groups = {}

          # get kappa_t or seconds For each experiment
          if kappa_t_OR_seconds
            g.keys.each{|k| groups[k] = g[k].map{|v| v[1].scan(kappa_or_seconds_regex).flatten[2]}.map &:to_f}
          else
            g.keys.each{|k| groups[k] = g[k].map{|v| v[1].scan(kappa_or_seconds_regex)}.flatten.map &:to_f}
          end

          # average over datasets
          if files_to_search.count > 2
            groups.keys.each {|k| groups[k] = groups[k].sum / groups[k].count}
          end
          results[file_to_search.gsub('|', ' ')] = groups
        end

        if results.empty?
          next
        end

        # save to CSV
        data, header = [], results[results.keys.first].keys.prepend('params \ dataset')
        results.keys.each {|key| data << results[key].values.prepend(key) }

        dir = "#{top_analysis_folder}/#{key.to_s}/#{subkey}/"
        FileUtils.mkdir_p(dir)
        csv_path = dir + measure + '.csv'
        CSV.open(csv_path, "wb") do |csv|
          csv << header
          for row in data
            csv << row
          end
        end

        # run Python program to evaluate statistical significance using numpy and external libraries
        if results.values.map(&:empty?).flatten.any?(false)
          py_params = {key: key, subkey: subkey, measure: measure, dir: dir}
          to_be_JSONed << {params: py_params, results: results}
        end
      end
    end
  end
  File.write("#{top_analysis_folder}/python.json", to_be_JSONed.to_json)
end

def part1
  top_dir = './experiment_results_step1'
  params = {
    window_type: ["SLIDING", "TUMBLING", "HYBRID"],
    voting_type: ["BOOLEAN", "PROBABILITY", "AVG_W_PROBABILITY", "W_AVG_PROBABILITY"],
    batch_size: [5,10,25,50,75,100],
    ground_truth: [100],
  }
  compute_statistical_significance(params, top_dir, :ground_truth)
end
  
def part2
  top_dir = 'experiment_results_step2_gt'
  params = {
    window_type: ["SLIDING", "HYBRID", "TUMBLING"],
    voting_type: ["AVG_W_PROBABILITY", "W_AVG_PROBABILITY"],
    batch_size: [25,50,75,100],
    ground_truth: [100,90,80,70,60],
  }
  compute_statistical_significance(params, top_dir, :none)
end

def part3(ground_truth, reject_key)
  top_dir = 'experiment_results_step3_drift'
  params = {
    window_type: ["SLIDING", "HYBRID"],
    voting_type: ["AVG_W_PROBABILITY", "W_AVG_PROBABILITY", "PROBABILITY", "BOOLEAN"],
    batch_size: [25,75,100],
    ground_truth: ground_truth,
    drift_reset_type: ['ALL', 'PARTIAL', 'BLIND_RANDOM'],
    drift_content: ['PROBABILITY', 'WEIGHTED_PROBABILITY', 'BOOLEAN'],
    drift_detector_count: ['ONE_PER_CLASSIFIER', 'ONE_FOR_ENSEMBLE'],
  }
  compute_statistical_significance(params, top_dir, reject_key)
end

# part1()
# part2()
# part3([100,90,80,70,60], :none)
part4()