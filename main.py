from multiprocessing import Process, Queue

import numpy as np

from loadgen.Infer_engine import InferEngine
from loadgen.load_generator import load_generator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastRec:
    def __init__(self, args):
        self.args = args

    def run(self):
        for key in vars(self.args):
            print(key, getattr(self.args, key))

        if self.args.queue:
            logger.info("Running in queue mode.")
            request_queue = Queue(maxsize=1024)
            # pid_queue = Queue()
            response_queue_list = [Queue() for _ in range(self.args.inference_engines)]
            inference_engine_ready_queue = Queue()

            # TODO： create a load generator
            load_generator_return_queue = Queue()
            loadgen = Process(target=load_generator,
                              args=(self.args, request_queue, load_generator_return_queue,
                                    inference_engine_ready_queue))

            engine_list = []
            for i in range(self.args.inference_engines):
                inf_engine = InferEngine(self.args)
                p = Process(target=inf_engine.run,
                            args=(request_queue, i, response_queue_list[i], inference_engine_ready_queue))
                p.daemon = True
                engine_list.append(p)

            logger.info("Starting inference engines...")
            for p in engine_list:
                p.start()

            # TODO: start the load generator
            logger.info("Starting load generator.")
            loadgen.start()

            response_list = []
            inference_engine_finished_num = 0
            response_dict = dict()
            response_latency, final_latencies = [], []

            granularity = int(self.args.req_granularity)

            logger.info("Monitoring responses.")
            while inference_engine_finished_num != self.args.inference_engines:
                for i in range(self.args.inference_engines):
                    if response_queue_list[i].qsize():
                        response = response_queue_list[i].get()

                        if response is None:
                            inference_engine_finished_num += 1
                            logger.info(f"Finished one inference engine. "
                                        f"Total:{inference_engine_finished_num}/{self.args.inference_engines}")
                        else:
                            # TODO: I think the exp packet in key can be removed
                            key = (response.epoch, response.batch_id, response.exp_packet)
                            if key in response_dict:
                                prev_value = response_dict[key]

                                arrival_time = min(response.arrival_time, prev_value[0])
                                inference_end_time = max(response.inference_end_time, prev_value[1])
                                remain_batches = response.total_sub_batches - 1
                                response_dict[key] = (arrival_time, inference_end_time, remain_batches)
                            else:
                                response_dict[key] = (response.arrival_time,
                                                      response.inference_end_time,
                                                      response.total_sub_batches - 1)

                            # it is the last one, so the whole request is already handled
                            if response.total_sub_batches == 1:
                                response_latency.append(response_dict[key][1] - response_dict[key][0])

                            if len(response_latency) % granularity:
                                logger.info(f"Running p95: "
                                            f"{np.percentile(response_latency[int(-1*granularity):], 95) * 1000.}")

                            response_list.append(response.__dict__)

            loadgen.join()
            total_requests = load_generator_return_queue.get()

            sub_requests_num, requests_num = total_requests
            logger.info(f"Total sub requests: {sub_requests_num}")
            logger.info(f"Total requests: {requests_num}")

            queries = list(filter(lambda x: x['sub_id'] == 0, response_list))
            start_time = queries[0]['inference_end_time']
            end_time = queries[-1]['inference_end_time']
            logger.info(f"QPS: {len(queries) / (end_time - start_time)}")
            logger.info(f"p95 tail-latency: {np.percentile(response_latency, 95) * 1000.} ms")
            logger.info(f"p99 tail-latency: {np.percentile(response_latency, 99) * 1000.} ms")

            for p in engine_list:
                p.terminate()
        else:
            infer_engine = InferEngine(self.args)
            infer_engine.run()


if __name__ == '__main__':
    from utils import cli
    args = cli()

    fastrec = FastRec(args)
    fastrec.run()
