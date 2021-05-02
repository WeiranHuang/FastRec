# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the dlrm benchmark
# The inpts and outputs are generated according to the following three option(s)
# 1) random distribution
# 2) synthetic distribution, based on unique accesses and distances between them
#    i) R. Hassan, A. Harris, N. Topham and A. Efthymiou "Synthetic Trace-Driven
#    Simulation of Cache Memory", IEEE AINAM'07
# 3) public data set
#    i) Kaggle Display Advertising Challenge Dataset
#     https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/


# others
import collections

from loadgen.data_generator import DataGenerator

from loadgen.dlrm_utils import *


class DLRMDataGenerator(DataGenerator):
    def __init__(self, args):
        super(DLRMDataGenerator,self).__init__(args)

    def generate_input_data(self):
        ln_bot = np.fromstring(self.args.arch_mlp_bot, dtype=int, sep="-")
        if self.args.data_generation == "dataset":
            # currently not implement the real dataset generator
            sys.exit("ERROR: Dataset based DLRM data generator is currently not supported")

        elif self.args.data_generation == "random":
            ln_emb = np.fromstring(self.args.arch_embedding_size, dtype=int, sep="-")
            m_den = ln_bot[0]
            (nbatches, lX, lS_l, lS_i) = self.generate_random_input_data(
                self.args.num_batches, self.args.max_mini_batch_size,
                self.args.round_targets, self.args.num_indices_per_lookup,
                self.args.num_indices_per_lookup_fixed, m_den, ln_emb)
            return (nbatches, lX, lS_l, lS_i)

        elif self.args.data_generation == "synthetic":
            ln_emb = np.fromstring(self.args.arch_embedding_size, dtype=int, sep="-")
            m_den = ln_bot[0]
            (nbatches, lX, lS_l, lS_i) = self.generate_synthetic_input_data(
                self.args.num_batches, self.args.max_mini_batch_size,
                self.args.round_targets, self.args.num_indices_per_lookup,
                self.args.num_indices_per_lookup_fixed, m_den, ln_emb,
                self.args.data_trace_file, self.args.data_trace_enable_padding)
            return (nbatches, lX, lS_l, lS_i)

        else:
            sys.exit("ERROR: --data-generation="
                     + self.args.data_generation + " is not supported")


    def generate_output_data(self):
        (nbatches, lT) = self.generate_random_output_data(
            self.args.num_batches, self.args.max_mini_batch_size,
            round_targets=self.args.round_targets)
        return (nbatches, lT)

    # uniform distribution (input data)
    @staticmethod
    def generate_random_input_data(num_batches, mini_batch_size, round_targets, num_indices_per_lookup,
                                   num_indices_per_lookup_fixed, m_den, ln_emb):
        nbatches = num_batches

        # inputs and targets
        lX = []
        lS_lengths = []
        lS_indices = []
        for j in range(0, nbatches):
            # number of data points in a batch
            n = mini_batch_size

            # dense feature
            Xt = ra.rand(n, m_den).astype(np.float32)
            lX.append(Xt)

            # sparse feature (sparse indices)
            lS_emb_lengths = []
            lS_emb_indices = []
            # for each embedding generate a list of n lookups,
            # where each lookup is composed of multiple sparse indices
            for size in ln_emb:
                lS_batch_lengths = []
                lS_batch_indices = []
                for _ in range(n):
                    # num of sparse indices to be used per embedding (between)
                    if num_indices_per_lookup_fixed:
                        sparse_group_size = np.int32(min(size, num_indices_per_lookup))
                    else:
                        r = ra.random(1)
                        sparse_group_size = np.int32(max(1, np.round(r * min(size, num_indices_per_lookup))[0]))

                    # sparse indices to be used per embedding
                    r = ra.random(sparse_group_size)
                    sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int32))

                    while sparse_group.size != sparse_group_size:
                        r = ra.random(sparse_group_size)
                        sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int32))

                    # reset sparse_group_size in case some index duplicates were removed
                    sparse_group_size = np.int32(sparse_group.size)
                    # store lengths and indices
                    lS_batch_lengths += [sparse_group_size]
                    lS_batch_indices += sparse_group.tolist()

                lS_emb_lengths.append(lS_batch_lengths)
                lS_emb_indices.append(lS_batch_indices)

            lS_lengths.append(lS_emb_lengths)
            lS_indices.append(lS_emb_indices)

        return (nbatches, lX, lS_lengths, lS_indices)


    # uniform distribution (output data)
    @staticmethod
    def generate_random_output_data(
            num_batches,
            mini_batch_size,
            num_targets=1,
            round_targets=False):

        nbatches = num_batches

        lT = []
        for j in range(0, nbatches):
            # number of data points in a batch
            n = mini_batch_size
            # target (probability of a click)
            if round_targets:
                P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.int32)
            else:
                P = ra.rand(n, num_targets).astype(np.float32)
            lT.append(P)

        return (nbatches, lT)


    # synthetic distribution (input data)
    @staticmethod
    def generate_synthetic_input_data(
            num_batches,
            mini_batch_size,
            round_targets,
            num_indices_per_lookup,
            num_indices_per_lookup_fixed,
            m_den,
            ln_emb,
            trace_file,
            enable_padding=False,
    ):
        nbatches = num_batches
        # print("Total number of batches %d" % nbatches)

        # inputs and targets
        lX = []
        lS_lengths = []
        lS_indices = []
        for j in range(0, nbatches):
            # number of data points in a batch
            n = mini_batch_size
            # dense feature
            Xt = ra.rand(n, m_den).astype(np.float32)
            lX.append(Xt)
            # sparse feature (sparse indices)
            lS_emb_lengths = []
            lS_emb_indices = []
            # for each embedding generate a list of n lookups,
            # where each lookup is composed of multiple sparse indices
            for i, size in enumerate(ln_emb):
                lS_batch_lengths = []
                lS_batch_indices = []
                for _ in range(n):
                    # num of sparse indices to be used per embedding (between
                    if num_indices_per_lookup_fixed:
                        sparse_group_size = np.int32(num_indices_per_lookup)
                    else:
                        # random between [1,num_indices_per_lookup])
                        r = ra.random(1)
                        sparse_group_size = np.int32(
                            max(1, np.round(r * min(size, num_indices_per_lookup))[0])
                        )
                    # sparse indices to be used per embedding
                    file_path = trace_file
                    line_accesses, list_sd, cumm_sd = read_dist_from_file(
                        file_path.replace("j", str(i))
                    )

                    r = trace_generate_lru(
                        line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
                    )

                    # WARNING: if the distribution in the file is not consistent with
                    # embedding table dimensions, below mod guards against out of
                    # range access
                    sparse_group = np.unique(r).astype(np.int32)
                    minsg = np.min(sparse_group)
                    maxsg = np.max(sparse_group)
                    if (minsg < 0) or (size <= maxsg):
                        print(
                            "WARNING: distribution is inconsistent with embedding "
                            + "table size (using mod to recover and continue)"
                        )
                        sparse_group = np.mod(sparse_group, size).astype(np.int32)
                    # sparse_group = np.unique(np.array(np.mod(r, size-1)).astype(np.int32))
                    # reset sparse_group_size in case some index duplicates were removed
                    sparse_group_size = np.int32(sparse_group.size)
                    # store lengths and indices
                    lS_batch_lengths += [sparse_group_size]
                    lS_batch_indices += sparse_group.tolist()
                lS_emb_lengths.append(lS_batch_lengths)
                lS_emb_indices.append(lS_batch_indices)
            lS_lengths.append(lS_emb_lengths)
            lS_indices.append(lS_emb_indices)

        return (nbatches, lX, lS_lengths, lS_indices)



if __name__ == "__main__":
    import sys
    sys.path.append("..")

    from utils import cli
    args = cli()
    print(args.num_batches)

    datagen = DLRMDataGenerator(args)
    (nbatches, lX, lS_l, lS_i) = datagen.generate_input_data()
    (nbatches, lT) = datagen.generate_output_data()

    print(nbatches)
    print(lX)
    print(lS_l)
    print(lS_i)
    # import sys
    # import operator
    # import argparse

    # ### parse arguments ###
    # parser = argparse.ArgumentParser(description="Generate Synthetic Distributions")
    # parser.add_argument("--trace-file", type=str, default="trace.log")
    # parser.add_argument("--trace-file-binary-type", type=bool, default=False)
    # parser.add_argument("--trace-enable-padding", type=bool, default=False)
    # parser.add_argument("--dist-file", type=str, default="dist.log")
    # parser.add_argument("--synthetic-file", type=str, default="trace_synthetic.log")
    # parser.add_argument("--numpy-rand-seed", type=int, default=123)
    # parser.add_argument("--print-precision", type=int, default=5)
    # args = parser.parse_args()

    # ### some basic setup ###
    # np.random.seed(args.numpy_rand_seed)
    # np.set_printoptions(precision=args.print_precision)

    # ### read trace ###
    # trace = read_trace_from_file(args.trace_file, args.trace_file_binary_type)
    # # print(trace)

    # ### profile trace ###
    # (_, stack_distances, line_accesses) = trace_profile(
    #     trace, args.trace_enable_padding
    # )
    # stack_distances.reverse()
    # line_accesses.reverse()
    # # print(line_accesses)
    # # print(stack_distances)

    # ### compute probability distribution ###
    # # count items
    # l = len(stack_distances)
    # dc = sorted(
    #     collections.Counter(stack_distances).items(), key=operator.itemgetter(0)
    # )

    # # create a distribution
    # list_sd = list(map(lambda tuple_x_k: tuple_x_k[0], dc))  # x = tuple_x_k[0]
    # dist_sd = list(
    #     map(lambda tuple_x_k: tuple_x_k[1] / float(l), dc)
    # )  # k = tuple_x_k[1]
    # cumm_sd = []  # np.cumsum(dc).tolist() #prefixsum
    # for i, (_, k) in enumerate(dc):
    #     if i == 0:
    #         cumm_sd.append(k / float(l))
    #     else:
    #         # add the 2nd element of the i-th tuple in the dist_sd list
    #         cumm_sd.append(cumm_sd[i - 1] + (k / float(l)))

    # ### write stack_distance and line_accesses to a file ###
    # write_dist_to_file(args.dist_file, line_accesses, list_sd, cumm_sd)

    # ### generate corresponding synthetic ###
    # # line_accesses, list_sd, cumm_sd = read_dist_from_file(args.dist_file)
    # synthetic_trace = trace_generate_lru(
    #     line_accesses, list_sd, cumm_sd, len(trace), args.trace_enable_padding
    # )
    # write_trace_to_file(args.synthetic_file, synthetic_trace, args.trace_file_binary_type)
