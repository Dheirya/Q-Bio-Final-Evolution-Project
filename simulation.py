import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True, linewidth=200)
rng = np.random.default_rng()

population_size = 100
lower_bound_gene = -1.0
upper_bound_gene = 1.0
num_genes = 10
lower_bound_gene_weight = 0.001
upper_bound_gene_weight = 0.999
trials = 10
population_limit = 10000


def index_value(individual_fitness):
  slope = 0.4
  midpoint = 0.2
  return 1 / (1 + np.exp(-slope * (individual_fitness - midpoint)))


def sim_experiment(mutation_rate):
  gene_weights = rng.uniform(lower_bound_gene_weight, upper_bound_gene_weight, num_genes)
  individuals = rng.uniform(-1.0, 1.0, (population_size, num_genes))
  overall_fitnesses = np.zeros(trials + 1)
  overall_fitnesses[0] = np.sum(individuals * gene_weights)
  population_counts = np.zeros(trials + 1)
  population_counts[0] = individuals.shape[0]
  average_fitnesses = np.zeros(trials + 1)
  average_fitnesses[0] = (np.sum(individuals * gene_weights) / individuals.shape[0])
  died = 0
  maxed = 0
  for k in range(1, trials + 1):
    i = 0
    while i < individuals.shape[0]:
      individuals[i] += rng.uniform(-mutation_rate, mutation_rate, num_genes)
      fitness_i = np.sum(individuals[i] * gene_weights)
      fitness_index = index_value(fitness_i)
      r = rng.random()
      if r < (1 - fitness_index):
        individuals = np.delete(individuals, i, axis=0)
        continue
      elif r < fitness_index and individuals.shape[0] < population_limit:
        individuals = np.vstack([individuals, individuals[i]])
      i += 1
    overall_fitnesses[k] = np.sum(individuals * gene_weights)
    population_counts[k] = individuals.shape[0]
    if individuals.shape[0] > 0:
      average_fitnesses[k] = (np.sum(individuals * gene_weights) / individuals.shape[0])
    else:
      average_fitnesses[k] = 0
    if individuals.shape[0] >= population_limit:
      overall_fitnesses = overall_fitnesses[:k+1]
      population_counts = population_counts[:k+1]
      average_fitnesses = average_fitnesses[:k+1]
      maxed = 1
      break
    elif individuals.shape[0] == 0:
      overall_fitnesses = overall_fitnesses[:k+1]
      population_counts = population_counts[:k+1]
      average_fitnesses = np.zeros(k+1)
      died = 1
      break
  result_matrix = np.vstack([overall_fitnesses, population_counts, average_fitnesses])
  return {"result": result_matrix, "dead": died, "maxed": maxed}


def multiple_experiments_avg(number, mutation_rate):
    results = []
    totDead = 0
    totMaxed = 0
    for _ in range(number):
        result_i = sim_experiment(mutation_rate)
        results.append(result_i["result"])
        totDead += result_i["dead"]
        totMaxed += result_i["maxed"]
    max_len = max(r.shape[1] for r in results)
    sum_overall = np.zeros(max_len)
    sum_population = np.zeros(max_len)
    sum_avgfit = np.zeros(max_len)
    counts = np.zeros(max_len)
    for r in results:
        L = r.shape[1]
        sum_overall[:L] += r[0]
        sum_population[:L] += r[1]
        sum_avgfit[:L] += r[2]
        counts[:L] += 1
    avg_overall = sum_overall / counts
    avg_population = sum_population / counts
    avg_avgfit = sum_avgfit / counts
    trial_averages = np.vstack([avg_overall, avg_population, avg_avgfit])
    sum_row_means = np.zeros(3)
    for r in results:
        sum_row_means += np.mean(r, axis=1)
    overall_averages = sum_row_means / number
    return {"trial_averages": trial_averages, "overall_averages": overall_averages, "change_fitness_average": round(float((trial_averages[2][10] - trial_averages[2][1]) / abs(trial_averages[2][1]) * 100), 2), "tot_dead": totDead, "tot_maxed": totMaxed}


def full_experiment(experiments):
  for MR in np.linspace(0.05, 0.5, 10):
    print(f"\nTRIAL (MR={MR})")
    data = multiple_experiments_avg(128, MR)
    trials = np.arange(data["trial_averages"].shape[1])
    labels = ["Total Fitness", "Total Population", "Individual Fitness"]
    ylabels = ["Total Fitness", "Total Population", "Individual Fitness"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
      axes[i].plot(trials, data["trial_averages"][i])
      axes[i].set_xlabel("Trial")
      axes[i].set_ylabel(ylabels[i])
      axes[i].set_title(labels[i])
      axes[i].grid(True)
    plt.tight_layout()
    plt.show()
    print(f"CHANGE IN FITNESS: %{data["change_fitness_average"]}, TOTAL MAXED POPS: {data["tot_maxed"]}, TOTAL DEAD POPS: {data["tot_dead"]}")
    print(f"\nAVG OVERALL FITNESS: {round(data["overall_averages"][0], 2)}, AVG POPULATION: {round(data["overall_averages"][1], 2)}, AVG INDIV FITNESS: {round(data["overall_averages"][2], 2)}")


full_experiment(128)
