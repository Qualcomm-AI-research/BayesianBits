# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.


from collections import OrderedDict
import torch


DEFAULT_BATCH_SIZE = 1
DEFAULT_INPUT_CHANNELS = 3
DEFAULT_RESOLUTION = [224, 224]
DEFAULT_INPUT_SHAPE = [DEFAULT_BATCH_SIZE, DEFAULT_INPUT_CHANNELS] + DEFAULT_RESOLUTION


class GraphMeta(object):
    def __init__(self, model_name=""):
        self._model_name = model_name
        self._nodes_by_name = OrderedDict()
        self._invocations = list()

    def add_node(self, full_name, layer, name, type, parms):
        """Register an operation node in the graph; any node can be added only once.
        Nodes are stored in the order they are registered."""
        if full_name in self._nodes_by_name:
            raise AssertionError("%s is an existing node" % full_name)
        self._nodes_by_name[full_name] = {
            "layer": layer,
            "name": name,
            "type": type,
            "parms": parms,
            "uses": 0,
            "macs": 0,
        }

    def add_invocation(self, node_name, macs):
        """Register an invocation of a registered operation node and store the
        specified number of MAC's. Nodes may be invoked multiple times.
        Invocations are stored in order."""
        node = self._nodes_by_name[node_name]
        node["uses"] += 1
        node["macs"] += macs  # we accumulate mac's across multiple invocations
        self._invocations.append(node_name)

    def add_winnow_info(self, node_name, inputs_to_ignore, outputs_to_ignore):
        node = self._nodes_by_name[node_name]
        node["void-inputs"] = inputs_to_ignore  # indices of input channels
        node["void-outputs"] = outputs_to_ignore  # indices of output channels

    @property
    def model_name(self):
        return self._model_name

    @property
    def num_nodes(self):
        return len(self._nodes_by_name)

    @property
    def num_parameters(self):
        return sum(node["parms"] for node in self._nodes_by_name.values())

    @property
    def num_invocations(self):
        return len(self._invocations)

    @property
    def num_macs(self):
        return sum(node["macs"] for node in self._nodes_by_name.values())

    def yield_node_names(self):
        """Return a generator on node names in the order they were registered."""
        for name in self._nodes_by_name.keys():
            yield name

    def yield_node_names_invoked(self, no_dups=False):
        """Return a generator on node names in the order nodes were invoked;
        nodes may occur multiple times unless 'no_dups' is True in which case
        only every first invocation is returned."""
        yielded = set()
        for name in self._invocations:
            if no_dups:
                if name in yielded:
                    continue
                yielded.add(name)
            yield name

    def get_stats_by_type(self):
        """Return statistics by node type: number of nodes in the graph,
        number of parameters, number of invocations, number of MAC's.
        Note that number of invocations is already included in number of MAC's """
        types = dict()
        for node_dict in self._nodes_by_name.values():
            typ = node_dict["type"]
            if typ not in types:
                types[typ] = {"nodes": 0, "parms": 0, "uses": 0, "macs": 0}
            type_dict = types[typ]
            type_dict["nodes"] += 1
            type_dict["parms"] += node_dict["parms"]
            type_dict["uses"] += node_dict["uses"]
            type_dict["macs"] += node_dict["macs"]
        return types

    def get_node_stats(self, node_name):
        """Return number of parameters, invocations and MAC's for the specified node."""
        return self._nodes_by_name[node_name]

    def dump(self):
        num_parms, num_invocations, num_macs, type_lines = self._gather_data_to_dump()

        print()
        print(
            "Graph consists of %d nodes and %d individual parameter values."
            % (self.num_nodes, num_parms)
        )
        print(
            "One forward run results in %d invocations and %d MAC's.\n"
            % (num_invocations, num_macs)
        )

        print("Instances, parameters, invocations and MAC's by node type:\n")
        for line in type_lines:
            print(line)
        print("               --- -------- -- ----------")
        print(
            "       total : %3d %8d %2d %10d"
            % (self.num_nodes, num_parms, num_invocations, num_macs)
        )
        print()

        print("Name, type, parameters, invocations and MACs by node invoked:\n")
        for name in self.yield_node_names_invoked(no_dups=True):
            stats = self.get_node_stats(name)
            print(
                "%32s : %12s %8d %2d %10d"
                % (name, stats["type"], stats["parms"], stats["uses"], stats["macs"])
            )
        print("                                   ------------ -------- -- ----------")
        print(
            "                           total : %12d %8d %2d %10d"
            % (self.num_nodes, num_parms, num_invocations, num_macs)
        )
        print()

    def _gather_data_to_dump(self):
        num_parms = 0
        num_invocations = 0
        num_macs = 0
        type_lines = []
        for typ, stats in self.get_stats_by_type().items():
            num_parms += stats["parms"]
            num_invocations += stats["uses"]
            num_macs += stats["macs"]
            type_lines.append(
                "%12s : %3d %8d %2d %10d"
                % (typ, stats["nodes"], stats["parms"], stats["uses"], stats["macs"])
            )
        return num_parms, num_invocations, num_macs, type_lines


class GraphInspector(object):
    """Inspects a computational graph and collects meta-data."""

    def __init__(
        self, module: torch.nn.Module, model_name="", input_shape: list = None
    ):
        """Constructor.
        #  :param input_shape - Used for single forward run as part of inspection"""
        self._start_module = module
        self._model_name = model_name
        self._input_shape = input_shape if input_shape else DEFAULT_INPUT_SHAPE
        self._module_info = dict()
        self._graph = None
        self._done = False
        self._added_hooks = []

    # noinspection PyUnusedLocal
    def inspect(self):
        """Inspect the implicit computational graph for the model and
        return an object containing meta-data so gathered."""

        if self._done:
            # don't bother with re-using an instance
            raise RuntimeError("Instance is dirty and cannot inspect again")

        model = self._start_module

        # empty list of hooks that were added
        self._added_hooks = []

        # gather static graph structure, and register forward() hook for each module
        meta = self._static_scan(model, self._model_name)

        # perform a single forward run; forward() hooks will record data in meta object
        model.eval()
        dummy_input = torch.rand(self._input_shape).to(next(model.parameters()).device)
        model(dummy_input)

        # remove all added hooks
        self._remove_all_hooks()

        self._done = True
        return meta

    def _remove_all_hooks(self):
        for hook in self._added_hooks:
            hook.remove()

    def _static_scan(self, model, model_name):
        """Construct hierarchy of modules and register forward() hook for each."""
        meta = GraphMeta(model_name)
        name_prefix = (model_name if model_name else type(model).__name__).lower()
        for full_name, module in model.named_modules(prefix=name_prefix):
            scope, name = self._split_full_name(full_name)
            num_parm_values = self._count_parm_values(module)
            meta.add_node(
                full_name, scope, name, type(module).__name__, num_parm_values
            )
            module_info = {"name": full_name, "obj": module}
            self._module_info[module] = module_info
            self._register_hook(module, meta)
        return meta

    def _split_full_name(self, long_name):
        sep_pos = long_name.rfind(".")
        if sep_pos >= 0:
            layer, name = long_name[:sep_pos], long_name[sep_pos + 1 :]
        else:
            layer, name = "", long_name
        return layer, name

    def _count_parm_values(self, module):
        num_parm_values = 0
        for p_name, p_obj in module.named_parameters():
            if "." not in p_name:  # ignore parameters of submodules
                num_parm_values += p_obj.numel()
        return num_parm_values

    def _register_hook(self, module, meta):
        def forward_hook(module, inp, oup):
            """Plain function that redirects to member function."""
            self._forward_hook(module, inp, oup, meta)

        handle = module.register_forward_hook(forward_hook)
        self._added_hooks.append(handle)

    def _forward_hook(self, module, inp, oup, meta):
        """Hook to fire upon completing forward() for each module"""

        node_dict = self._module_info.get(module, None)
        full_name = node_dict["name"]
        num_macs = 0

        inp = inp[0] if isinstance(inp, tuple) and len(inp) == 1 else inp
        oup = inp[0] if isinstance(oup, tuple) and len(oup) == 1 else oup

        # module type specific inspection
        if isinstance(module, torch.nn.Conv2d):
            num_macs = self._assess_conv2d(module, inp, oup)
        elif isinstance(module, torch.nn.Linear):
            num_macs = self._assess_linear(module, inp, oup)
        elif isinstance(module, torch.nn.Conv1d):
            num_macs = self._assess_conv1d(module, inp, oup)

        meta.add_invocation(full_name, num_macs)

    def _assess_conv2d(self, module, inp, oup):
        shape_kernel = module.kernel_size
        if isinstance(shape_kernel, tuple):
            num_kernel_elem = shape_kernel[0] * shape_kernel[1]
        else:
            num_kernel_elem = shape_kernel ** 2
        shape_in = inp.shape
        batch_size = shape_in[0]
        channels_in = shape_in[1]
        assert channels_in % module.groups == 0
        shape_out = oup.shape
        assert shape_out[0] == batch_size
        channels_out = shape_out[1]
        assert channels_out % module.groups == 0
        num_out_elem = shape_out[2] * shape_out[3]
        num_macs = (
            (channels_out * num_out_elem)
            * num_kernel_elem
            * channels_in
            // module.groups
        )
        num_macs *= batch_size  # ignoring bias additions
        return num_macs

    def _assess_conv1d(self, module, inp, oup):
        shape_kernel = module.kernel_size
        num_kernel_elem = (
            shape_kernel[0] if isinstance(shape_kernel, tuple) else shape_kernel
        )
        shape_in = inp.shape
        batch_size = shape_in[0]
        channels_in = shape_in[1]
        assert channels_in % module.groups == 0
        shape_out = oup.shape
        assert shape_out[0] == batch_size
        channels_out = shape_out[1]
        assert channels_out % module.groups == 0
        num_out_elem = shape_out[2]
        num_macs = (
            (channels_out * num_out_elem)
            * num_kernel_elem
            * channels_in
            // module.groups
        )
        num_macs *= batch_size  # ignoring bias additions
        return num_macs

    def _assess_linear(self, module, inp, oup):
        features_in, features_out = inp.shape[-1], oup.shape[-1]
        num_macs = features_out * features_in  # ignoring bias additions
        for dim_size in inp.shape[
            :-1
        ]:  # includes batch size dimension i.e. the first dimension
            num_macs *= dim_size
        return num_macs


def inspect_graph(module: torch.nn.Module, model_name="", input_shape: list = None):
    """Convenience function wrapping GraphInspector."""
    return GraphInspector(module, model_name, input_shape).inspect()
