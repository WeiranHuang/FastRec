import sys
sys.path.append('..')

import threading
import time
from multiprocessing import Queue

import numpy as np
from caffe2.python import workspace

from loadgen.dlrm_datagen import DLRMDataGenerator
from models.dlrm_queue import DLRMWrapper
from loadgen.packet import ServiceResponse
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

def run_model(model, args, internal_logging, response_queue):
    """
    Get an entry from internal logging, after processing,
    put the response into the response queue.
    :param model:
    :param args:
    :param internal_logging:
    :param response_queue:
    :return:
    """
    top_fc_layers = args.arch_mlp_top.split("-")
    fc_tag = f"top:::fc{len(top_fc_layers) - 1}_z"
    while True:
        model.dlrm.run()
        response = internal_logging.get()
        if response is None:
            return

        end_time = time.time()
        out_size = np.array(workspace.FetchBlob(fc_tag)) / int(top_fc_layers[-2])
        response.inference_end_time = end_time
        response.out_batch_size = out_size
        response_queue.put(response)


class InferEngine(object):
    def __init__(self, args):
        self.args = args

    def run(self, request_queue=None, engine_id=None, response_queue=None, inferEngine_ready_queue=None):
        inference_logging = Queue()
        # inference_done = Queue()

        # utils
        np.random.seed(self.args.numpy_rand_seed)
        np.set_printoptions(precision=self.args.print_precision)

        # data
        datagen = DLRMDataGenerator(self.args)
        (nbatches, lX, lS_l, lS_i) = datagen.generate_input_data()
        (nbatches, lT) = datagen.generate_output_data()
        # logger.info(nbatches)
        # logger.info(lX)
        # logger.info(lS_l)
        # logger.info(lS_i)
        # print(nbatches)
        # print(lX)
        # print(lS_l)
        # print(lS_i)
        

        # model
        model = DLRMWrapper(self.args)
        model.create(lX[0], lS_l[0], lS_i[0], lT[0])

        # logger.info(lX[0], lS_l[0], lS_i[0], lT[0])

        # if no queue, directly feed the data generated to the model
        if request_queue is None:
            total_time = 0
            data_load_time = 0

            start_time = time.time()
            for i in range(self.args.nepochs):
                for j in range(nbatches):
                    start_load_time = time.time()
                    end_load_time = model.dlrm.run(lX[j], lS_l[j], lS_i[j])
                    data_load_time += (end_load_time - start_load_time)

            end_time = time.time()
            data_load_time *= 1000
            total_time += (end_time - start_time) * 1000
            
            # logger.info(f"Total data loading time: {data_load_time} ms")
            # logger.info(f"Total data loading time: {data_load_time / (args.nepochs * nbatches)} ms/iter")
            # logger.info(f"Total computation time: {(total_time - data_load_time)} ms")
            # logger.info(f"Total computation time: {(total_time - data_load_time) / (args.nepochs * nbatches)} ms/iter")
            # logger.info(f"Total execution time: {total_time} ms")
            # logger.info(f"Total execution time: {total_time / (args.nepochs * nbatches)} ms/iter")
        else:
            #
            inference_thread = threading.Thread(target=run_model,
                                                args=(model, self.args, inference_logging, response_queue))
            inference_thread.daemon = True  # always running
            inference_thread.start()

            while True:
                inferEngine_ready_queue.put(True)
                request = request_queue.get()

                if request is None:
                    time.sleep(4)
                    inference_logging.put(None)
                    response_queue.put(None)
                    return

                batch_id = request.batch_id
                lS_l_cur = np.transpose(np.array(lS_l[batch_id]))
                lS_l_cur = np.transpose(np.array(lS_l_cur[:request.batch_size]))
                lS_i_cur = np.array(lS_i[batch_id])
                lS_i_cur = np.array(
                    lS_i_cur[:][:, :request.batch_size * self.args.num_indices_per_lookup])  # todo: make it not fixed
                # print('------')
                # print(lS_l_cur)
                # print(lS_i_cur)

                start_time = time.time()
                model.run_queues(lS_i_cur, lS_l_cur, lX[batch_id][:request.batch_size], request.batch_size)
                end_time = time.time()
                response = ServiceResponse(consumer_id=engine_id,
                                           epoch=request.epoch,
                                           batch_id=request.batch_id,
                                           batch_size=request.batch_size,
                                           arrival_time=request.batch_size,
                                           process_start_time=start_time,
                                           queue_end_time=end_time,
                                           total_sub_batches=request.total_sub_batches,
                                           exp_packet=request.request_id,
                                           sub_id=request.sub_id)

                print("Generate one response:")
                print(response)

                inference_logging.put(response)
        return


if __name__ == '__main__':
    from utils import cli

    args = cli()

    inf_engine = InferEngine(args)
    inf_engine.run()
