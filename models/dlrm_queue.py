from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python._import_c_extension as C

from models.dlrm import DLRMNet

# define a wrapper for DLRM to decouple input queues and DLRMNet's logic.
class DLRMWrapper():
    def FeedBlobWrapper(self, tag, val):
        if self.accel_en:
            _d = core.DeviceOption(caffe2_pb2.CUDA, 0)
            with core.DeviceScope(_d):
                workspace.FeedBlob(tag, val, device_option=_d)
        else:
            workspace.FeedBlob(tag, val)

    @staticmethod
    def build_dlrm_mlp_queue():
        mlp_q_net = core.Net("fc_q_init")

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            mlp_queue = mlp_q_net.CreateBlobsQueue([], "fc_q_blob", num_blobs=1, capacity=8)

        workspace.RunNetOnce(mlp_q_net)

        mlp_input_net = core.Net("fc_input_net")
        mlp_input_net.EnqueueBlobs([mlp_queue, "fc_inputs"], ["fc_inputs"])

        return mlp_queue, "fc_inputs", mlp_input_net


    @staticmethod
    def build_dlrm_emb_queue(tag="id", qid=None):
        emb_q_net = core.Net(tag + '_q_init_' + str(qid))

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            emb_queue = emb_q_net.CreateBlobsQueue([], tag + "_q_blob_" + str(qid))

        workspace.RunNetOnce(emb_q_net)

        emb_input_net = core.Net(tag + "_input_net_" + str(qid))
        emb_input_net.EnqueueBlobs([emb_queue, tag + "_inputs_" + str(qid)], [tag + "_inputs_" + str(qid)])

        return emb_queue, tag + "_inputs_" + str(qid), emb_input_net

    def __init__(self,
                 cli_args,
                 model=None,
                 tag=None,
                 enable_prof=False):
        self.args = cli_args
        self.accel_en = self.args.use_accel

        if self.accel_en:
            device_opt = core.DeviceOption(caffe2_pb2.CUDA, 0)
            n_accels = C.num_cuda_devices
            print(f"(Wrapper) Using {n_accels} Accel(s)...")
        else:
            device_opt = core.DeviceOption(caffe2_pb2.CPU)
            print("(Wrapper) Using CPU...")

        num_tables = len(cli_args.arch_embedding_size.split('-'))

        self.id_qs, self.id_input_blobs, self.id_input_nets = [], [], []
        self.len_qs, self.len_input_blobs, self.len_input_nets = [], [], []

        for i in range(num_tables):
            q, input_blob, net = self.build_dlrm_emb_queue(tag="id", qid=i)
            self.id_qs.append(q)
            self.id_input_blobs.append(input_blob)
            self.id_input_nets.append(net)

            q, input_blob, net = self.build_dlrm_emb_queue(tag="len", qid=i)
            self.len_qs.append(q)
            self.len_input_blobs.append(input_blob)
            self.len_input_nets.append(net)

        self.fc_q, self.fc_input_blob, self.fc_input_net = self.build_dlrm_mlp_queue()

        if self.args.queue:
            with core.DeviceScope(device_opt):
                self.dlrm = DLRMNet(cli_args, model, tag, enable_prof,
                                     id_qs = self.id_qs,
                                     len_qs = self.len_qs,
                                     fc_q   = self.fc_q)
        else:
            with core.DeviceScope(device_opt):
                self.dlrm = DLRMNet(cli_args, model, tag, enable_prof)


    def create(self, X, S_lengths, S_indices, T):
        if self.args.queue:
            self.dlrm.create(X, S_lengths, S_indices, T,
                             id_qs = self.id_qs,
                             len_qs = self.len_qs)
        else:
            self.dlrm.create(X, S_lengths, S_indices, T)


    # Run the Queues to provide inputs to DLRM model
    def run_queues(self, ids, lengths, fc, batch_size):
        # Dense features
        self.FeedBlobWrapper(self.fc_input_blob, fc)
        workspace.RunNetOnce(self.fc_input_net.Proto())

        # Sparse features
        num_tables = len(self.args.arch_embedding_size.split("-"))
        for i in range(num_tables):
            self.FeedBlobWrapper(self.id_input_blobs[i], ids[i])
            workspace.RunNetOnce(self.id_input_nets[i].Proto())

            self.FeedBlobWrapper(self.len_input_blobs[i], lengths[i])
            workspace.RunNetOnce(self.len_input_nets[i].Proto())


if __name__ == '__main__':
    queue, name, net = DLRMWrapper.build_dlrm_mlp_queue()
    print(queue._from_net, name, net.Proto())
    queue, name, net = DLRMWrapper.build_dlrm_emb_queue(tag="id", qid=1)
    print(queue._from_net, name, net.Proto())


