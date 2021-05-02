# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereia, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

import time
import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import core, model_helper, workspace, net_drawer

import sys


class DLRMNet(object):
    def FeedBlobWrapper(self, tag, val):
        """
        Wrap the process of feeding the blob into the workspace.
        If accelerator is enabled, use it.
        :param tag: The name of the blob.
        :param val: either a TensorProto object or a numpy array object to be fed into
          the workspace.
        :return:
        """
        if self.accel_en:
            _d = core.DeviceOption(caffe2_pb2.CUDA, 0)
            # with core.DeviceScope(_d):
            workspace.FeedBlob(tag, val, device_option=_d)
        else:
            workspace.FeedBlob(tag, val)

    def create_mlp(self, ln, sigmoid_layer, model, tag, fc_q=None):
        """

        :param ln: a list of dimensions.
        :param sigmoid_layer: sigmod or relu.
        :param model: The DLRM model.
        :param tag: (layer's tag, input's tag, output's tag).
        :param fc_q: the dense features in queue.
        :return:
        """
        (tag_layer, tag_in, tag_out) = tag

        # build MLP layer by layer
        layers = []
        weights = []
        for i in range(1, ln.size):
            n = ln[i - 1]
            m = ln[i]

            # create tags
            tag_fc_w = tag_layer + ":::" + "fc" + str(i) + "_w"
            tag_fc_b = tag_layer + ":::" + "fc" + str(i) + "_b"
            tag_fc_y = tag_layer + ":::" + "fc" + str(i) + "_y"
            tag_fc_z = tag_layer + ":::" + "fc" + str(i) + "_z"
            if i == ln.size - 1:
                tag_fc_z = tag_out
            weights.append(tag_fc_w)
            weights.append(tag_fc_b)

            # initialize the weights with xavier
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)

            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            b = np.random.normal(mean, std_dev, size=m).astype(np.float32)

            self.FeedBlobWrapper(tag_fc_w, W)
            self.FeedBlobWrapper(tag_fc_b, b)

            # approach 1: construct fully connected operator using model.net
            if self.args.queue and (fc_q is not None) and (i == 1):
                # Dequeue lengths vector as well
                model.net.DequeueBlobs(fc_q, tag_in)
                fc = model.net.FC([tag_in, tag_fc_w, tag_fc_b], tag_fc_y,
                                  engine=self.args.engine,
                                  max_num_tasks=self.args.fc_workers)
            else:
                fc = model.net.FC([tag_in, tag_fc_w, tag_fc_b], tag_fc_y,
                                  engine=self.args.engine,
                                  max_num_tasks=self.args.fc_workers)

            layers.append(fc)

            if i == sigmoid_layer:
                layer = model.net.Sigmoid(tag_fc_y, tag_fc_z)

            else:
                layer = model.net.Relu(tag_fc_y, tag_fc_z)
            tag_in = tag_fc_z
            layers.append(layer)

        # WARNING: the dependency between layers is implicit in the tags,
        # so only the last layer is added to the layers list. It will
        # later be used for interactions.
        return layers, weights

    def create_emb(self, m, ln, model, tag, id_qs=None, len_qs=None):
        (tag_layer, tag_in, tag_out) = tag
        emb_l = []
        weights_l = []
        for i in range(0, ln.size):
            n = ln[i]

            # create tags
            len_s = tag_layer + ":::" + "sls" + str(i) + "_l"
            ind_s = tag_layer + ":::" + "sls" + str(i) + "_i"
            tbl_s = tag_layer + ":::" + "sls" + str(i) + "_w"
            sum_s = tag_layer + ":::" + "sls" + str(i) + "_z"
            weights_l.append(tbl_s)

            # initialize the weights
            # approach 1a: custom
            W = np.random.uniform(low=-np.sqrt(1 / n),
                                  high=np.sqrt(1 / n),
                                  size=(n, m)).astype(np.float32)

            self.FeedBlobWrapper(tbl_s, W)
            if self.args.queue:
                # If want to have non-blocking IDs we have to dequeue the input
                # ID blobs on the model side
                model.net.DequeueBlobs(id_qs[i], ind_s + "_pre_cast")
                model.net.Cast(ind_s + "_pre_cast", ind_s,
                               to=core.DataType.INT32)
                # Operator Mod is not found in Caffe2 latest build
                # model.net.Mod(ind_s + "_pre_mod", ind_s, divisor = n)

                # Dequeue lengths vector as well
                model.net.DequeueBlobs(len_qs[i], len_s)

            # create operator
            if self.accel_en:
                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
                    EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s],
                                                    engine=self.args.engine,
                                                    max_num_tasks=self.args.sls_workers)
            else:
                EE = model.net.SparseLengthsSum([tbl_s, ind_s, len_s], [sum_s],
                                                engine=self.args.engine,
                                                max_num_tasks=self.args.sls_workers)

            emb_l.append(EE)

        return emb_l, weights_l

    def create_interactions(self, x, ly, model, tag):
        (tag_dense_in, tag_sparse_in, tag_int_out) = tag

        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            tag_int_out_info = tag_int_out + "_info"
            T, T_info = model.net.Concat(
                x + ly,
                [tag_int_out + "_cat_axis0", tag_int_out_info + "_cat_axis0"],
                axis=1,
                add_axis=1,
            )
            # perform a dot product
            Z = model.net.BatchMatMul([T, T], tag_int_out + "_matmul", trans_b=1)
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = model.net.Flatten(Z, tag_int_out + "_flatten", axis=1)
            # approach 2: unique
            Zflat_all = model.net.Flatten(Z, tag_int_out + "_flatten_all", axis=1)
            Zflat = model.net.BatchGather([Zflat_all, tag_int_out + "_tril_indices"],
                                          tag_int_out + "_flatten")
            R, R_info = model.net.Concat(
                x + [Zflat], [tag_int_out, tag_int_out_info], axis=1
            )
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            tag_int_out_info = tag_int_out + "_info"
            R, R_info = model.net.Concat(
                x + ly, [tag_int_out, tag_int_out_info], axis=1
            )
        else:
            sys.exit("ERROR: --arch-interaction-op="
                     + self.arch_interaction_op + " is not supported")

        return R

    def create_sequential_forward_ops(self, id_qs=None, len_qs=None, fc_q=None):
        # embeddings
        tag = (self.temb, self.tsin, self.tsout)
        self.emb_l, self.emb_w = self.create_emb(self.m_spa, self.ln_emb,
                                                 self.model, tag,
                                                 id_qs=id_qs,
                                                 len_qs=len_qs)
        # bottom mlp
        tag = (self.tbot, self.tdin, self.tdout)
        self.bot_l, self.bot_w = self.create_mlp(self.ln_bot, self.sigmoid_bot,
                                                 self.model, tag, fc_q=fc_q)
        # interactions
        tag = (self.tdout, self.tsout, self.tint)
        Z = self.create_interactions([self.bot_l[-1]], self.emb_l, self.model, tag)

        # top mlp
        tag = (self.ttop, Z, self.tout)
        self.top_l, self.top_w = self.create_mlp(self.ln_top, self.sigmoid_top,
                                                 self.model, tag
                                                 )

        # setup the last output variable
        self.last_output = self.top_l[-1]

    def __init__(
            self,
            cli_args,
            model=None,
            tag=None,
            enable_prof=False,
            id_qs=None,
            len_qs=None,
            fc_q=None
    ):
        """
        :param cli_args: Command line arguments.
        :param model: The DLRM model.
        :param tag: name tags of layers.
        :param enable_prof: Profiling flag.
        :param id_qs: id queue.
        :param len_qs: len queue.
        :param fc_q: fc queue.
        """
        super(DLRMNet, self).__init__()
        self.args = cli_args

        ### parse command line arguments ###
        ln_bot = np.fromstring(cli_args.arch_mlp_bot, dtype=int, sep="-")
        m_den = ln_bot[0]

        m_spa = cli_args.arch_sparse_feature_size
        ln_emb = np.fromstring(cli_args.arch_embedding_size, dtype=int, sep="-")
        num_fea = ln_emb.size + 1  # num sparse + num dense features
        m_den_out = ln_bot[ln_bot.size - 1]

        accel_en = self.args.use_accel

        if cli_args.arch_interaction_op == "dot":
            # approach 1: all
            # num_int = num_fea * num_fea + m_den_out
            # approach 2: unique
            if cli_args.arch_interaction_itself:
                num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
            else:
                num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
        elif cli_args.arch_interaction_op == "cat":
            num_int = num_fea * m_den_out
        else:
            sys.exit("ERROR: --arch-interaction-op="
                     + cli_args.arch_interaction_op + " is not supported")

        arch_mlp_top_adjusted = str(num_int) + "-" + cli_args.arch_mlp_top
        ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

        if m_den != ln_bot[0]:
            sys.exit("ERROR: arch-dense-feature-size "
                     + str(m_den) + " does not match first dim of bottom mlp " + str(ln_bot[0]))
        if m_spa != m_den_out:
            sys.exit("ERROR: arch-sparse-feature-size "
                     + str(m_spa) + " does not match last dim of bottom mlp " + str(m_den_out))
        if num_int != ln_top[0]:
            sys.exit("ERROR: # of feature interactions "
                     + str(num_int) + " does not match first dim of top mlp " + str(ln_top[0]))

        ### initialize the model ###
        if model is None:
            global_init_opt = ["caffe2", "--caffe2_log_level=0"]
            if enable_prof:
                global_init_opt += [
                    "--logtostderr=0",
                    "--log_dir=$HOME",
                    # "--caffe2_logging_print_net_summary=1",
                ]
            workspace.GlobalInit(global_init_opt)
            self.set_tags()
            self.model = model_helper.ModelHelper(name="DLRM", init_params=True)

            if cli_args:
                self.model.net.Proto().type = cli_args.caffe2_net_type
                self.model.net.Proto().num_workers = cli_args.inter_op_workers

        else:
            # WARNING: assume that workspace and tags have been initialized elsewhere
            self.set_tags(tag[0], tag[1], tag[2], tag[3], tag[4], tag[5], tag[6],
                          tag[7], tag[8], tag[9])
            self.model = model

        # save arguments
        self.m_spa = m_spa
        self.ln_emb = ln_emb
        self.ln_bot = ln_bot
        self.ln_top = ln_top
        self.arch_interaction_op = cli_args.arch_interaction_op
        self.arch_interaction_itself = cli_args.arch_interaction_itself
        self.sigmoid_bot = -1  # TODO: Lets not hard-code this going forward
        self.sigmoid_top = ln_top.size - 1
        self.accel_en = accel_en

        self.create_sequential_forward_ops(id_qs, len_qs, fc_q)

    def set_tags(
            self,
            _tag_layer_top_mlp="top",
            _tag_layer_bot_mlp="bot",
            _tag_layer_embedding="emb",
            _tag_feature_dense_in="dense_in",
            _tag_feature_dense_out="dense_out",
            _tag_feature_sparse_in="sparse_in",
            _tag_feature_sparse_out="sparse_out",
            _tag_interaction="interaction",
            _tag_dense_output="prob_click",
            _tag_dense_target="target",
    ):
        # layer tags
        self.ttop = _tag_layer_top_mlp
        self.tbot = _tag_layer_bot_mlp
        self.temb = _tag_layer_embedding
        # dense feature tags
        self.tdin = _tag_feature_dense_in
        self.tdout = _tag_feature_dense_out
        # sparse feature tags
        self.tsin = _tag_feature_sparse_in
        self.tsout = _tag_feature_sparse_out
        # output and target tags
        self.tint = _tag_interaction
        self.ttar = _tag_dense_target
        self.tout = _tag_dense_output

    def parameters(self):
        return self.model

    def create(self, X, S_lengths, S_indices, T, id_qs=None, len_qs=None):
        self.create_input(X, S_lengths, S_indices, T)
        self.create_model(X, S_lengths, S_indices, T)

    def create_input(self, X, S_lengths, S_indices, T):
        # feed input data to blobs
        self.FeedBlobWrapper(self.tdin, X)

        for i in range(len(self.emb_l)):
            len_s = self.temb + ":::" + "sls" + str(i) + "_l"
            ind_s = self.temb + ":::" + "sls" + str(i) + "_i"
            self.FeedBlobWrapper(len_s, np.array(S_lengths[i]))
            self.FeedBlobWrapper(ind_s, np.array(S_indices[i]))

        # feed target data to blobs
        if T is not None:
            zeros_fp32 = np.zeros(T.shape).astype(np.float32)
            self.FeedBlobWrapper(self.ttar, zeros_fp32)

    def create_model(self, X, S_lengths, S_indices, T):
        # setup tril indices for the interactions
        offset = 1 if self.arch_interaction_itself else 0
        num_fea = len(self.emb_l) + 1
        tril_indices = np.array([j + i * num_fea
                                 for i in range(num_fea) for j in range(i + offset)])
        self.FeedBlobWrapper(self.tint + "_tril_indices", tril_indices)

        # create compute graph
        print("Trying to run DLRM for the first time")
        sys.stdout.flush()
        if T is not None:
            # WARNING: RunNetOnce call is needed only if we use brew and ConstantFill.
            # We could use direct calls to self.model functions above to avoid it
            workspace.RunNetOnce(self.model.param_init_net)
            workspace.CreateNet(self.model.net)
        print("Ran DLRM for the first time")
        sys.stdout.flush()

    def run(self, X=None, S_lengths=None, S_indices=None, enable_prof=False):
        # feed input data to blobs
        if not self.args.queue:
            # dense features
            self.FeedBlobWrapper(self.tdin, X)

            # sparse features
            for i in range(len(self.emb_l)):
                ind_s = self.temb + ":::" + "sls" + str(i) + "_i"
                self.FeedBlobWrapper(ind_s, np.array(S_indices[i]))

                len_s = self.temb + ":::" + "sls" + str(i) + "_l"
                self.FeedBlobWrapper(len_s, np.array(S_lengths[i]))

        load_time = time.time()
        # execute compute graph
        if enable_prof:
            workspace.C.benchmark_net(self.model.net.Name(), 0, 1, True)
        else:
            workspace.RunNet(self.model.net)
        return load_time


if __name__ == "__main__":
    from utils import cli

    args = cli()
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
        dlrm = DLRMNet(args)

    print(dlrm.model.net.Proto())

    print(workspace.FetchBlob("top:::fc3_w"))

    # graph = net_drawer.GetPydotGraph(
    #     dlrm.parameters().net,
    #     "dlrm_s_caffe2_graph",
    #     "BT"
    # )
    #
    # graph.write_pdf(graph.get_name() + ".pdf")
