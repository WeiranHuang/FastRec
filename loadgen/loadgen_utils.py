import numpy as np
import time

# Possion distribution for arrival interval patterns
def model_arrival_times(args):
  arrival_time_delays = np.random.poisson(lam  = args.avg_arrival_rate,
                                          size = args.nepochs * args.num_batches)
  return arrival_time_delays


def model_batch_size_distribution(args):
  if args.batch_size_distribution == "normal":
    batch_size_distributions = np.random.normal(args.avg_mini_batch_size, args.var_mini_batch_size, args.num_batches)

  elif args.batch_size_distribution == "lognormal":
    batch_size_distributions = np.random.lognormal(args.avg_mini_batch_size, args.var_mini_batch_size, args.num_batches)

  elif args.batch_size_distribution == "fixed":
    batch_size_distributions = np.array([args.avg_mini_batch_size for _ in range(args.num_batches) ])

  elif args.batch_size_distribution == "file":
    percentiles = []
    batch_size_distributions = []
    with open(args.batch_dist_file, 'r') as f:
      lines = f.readlines()
      for line in lines:
        percentiles.append(float(line.rstrip()))

      for _ in range(args.num_batches):
        batch_size_distributions.append( int(percentiles[ int(np.random.uniform(0, len(percentiles))) ]) )

  for i in range(args.num_batches):
    batch_size_distributions[i] = int(max(min(batch_size_distributions[i], args.max_mini_batch_size), 1))
  return batch_size_distributions

# partition the requests into small batches
def partition_requests(args, batch_size):
  batch_sizes = []

  while batch_size > 0:
    mini_batch_size = min(args.sub_task_batch_size, batch_size)
    batch_sizes.append(mini_batch_size)
    batch_size -= mini_batch_size

  return batch_sizes


def loadGenSleep( sleeptime ):
  if sleeptime > 0.0055:
    time.sleep(sleeptime)
  else:
    startTime = time.time()
    while (time.time() - startTime) < sleeptime:
      continue
  return