from loadgen.loadgen_utils import *
from loadgen.packet import ServiceRequest


def load_generator(args, request_queue, load_generator_return_queue, inferEngine_ready_queue):
    ready_engines_num = 0

    # add engines as configured
    while ready_engines_num < args.inference_engines:
        inferEngine_ready_queue.get()
        ready_engines_num += 1

    arrival_time_delays = model_arrival_times(args)
    batch_size_distributions = model_batch_size_distribution(args)

    requests_num, sub_requests_num = 0, 0
 
    # generate the load
    for epoch in range(args.nepochs):
        for batch_id in range(args.num_batches):
            request_id = epoch * args.num_batches + batch_id
            request_size = int(batch_size_distributions[batch_id])  # have to round to integers

            # to see if we have to partition the request
            batch_sizes = partition_requests(args, request_size)
            for i, batch_size in enumerate(batch_sizes):
                request = ServiceRequest(request_id=request_id,
                                         batch_id=batch_id,
                                         epoch=epoch,
                                         batch_size=batch_size,
                                         sub_id=i,
                                         total_sub_batches=len(batch_sizes),
                                         )
                sub_requests_num += 1
                request.arrival_time = time.time()
                request_queue.put(request)
            requests_num += 1
            arrival_time = arrival_time_delays[request_id]
            loadGenSleep(arrival_time/1000.)

    for i in range(args.inference_engines):
        print(f"[Load Generator] Done sending signals to engine {i}.")
        request_queue.put(None)  # done flag

    load_generator_return_queue.put((sub_requests_num, requests_num))




