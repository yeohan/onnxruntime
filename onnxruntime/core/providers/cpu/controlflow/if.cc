// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/controlflow/if.h"
#include "core/providers/cpu/controlflow/utils.h"

#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

/*
ONNX_OPERATOR_SET_SCHEMA(
    If,
    1,
    OpSchema()
        .SetDoc("If conditional")
        .Input(0, "cond", "Condition for the if", "B")
        .Output(
            0,
            "outputs",
            "Values that are live-out to the enclosing scope. The return values in "
            "the `then_branch` and `else_branch` must be of the same shape and same "
            "data type.",
            "V",
            OpSchema::Variadic)
        .Attr(
            "then_branch",
            "Graph to run if condition is true. Has N outputs: values you wish to "
            "be live-out to the enclosing scope. The number of outputs must match"
            " the number of outputs in the else_branch.",
            AttributeProto::GRAPH)
        .Attr(
            "else_branch",
            "Graph to run if condition is false. Has N outputs: values you wish to"
            " be live-out to the enclosing scope. The number of outputs must match"
            " the number of outputs in the then_branch.",
            AttributeProto::GRAPH)
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
        .TypeConstraint("B", {"tensor(bool)"}, "Only bool"));
*/

ONNX_CPU_OPERATOR_KERNEL(If,
                         1,
                         KernelDefBuilder()
                             .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
                             .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                         If);

struct If::Info {
  Info(const onnxruntime::Node& node, const GraphViewer& subgraph_in) : subgraph{subgraph_in} {
    num_implicit_inputs = static_cast<int>(node.ImplicitInputDefs().size());
    used_implicit_inputs = std::vector<bool>(num_implicit_inputs, true);
    num_outputs = static_cast<int>(node.OutputDefs().size());

    auto& subgraph_outputs = subgraph.GetOutputs();
    auto num_subgraph_outputs = subgraph_outputs.size();

    ORT_ENFORCE(num_subgraph_outputs == static_cast<size_t>(num_outputs),
                "'If' node has ", num_outputs, " outputs which doesn't match the subgraph's ",
                num_subgraph_outputs, " outputs.");

    subgraph_output_names.reserve(num_subgraph_outputs);
    for (size_t i = 0; i < num_subgraph_outputs; ++i) {
      auto& output = subgraph_outputs[i];
      subgraph_output_names.push_back(output->Name());
    }
  }

  const GraphViewer& subgraph;

  std::vector<bool> used_implicit_inputs;
  int num_implicit_inputs;
  int num_outputs;

  std::vector<std::string> subgraph_output_names;
};

class IfImpl {
 public:
  IfImpl(OpKernelContextInternal& context,
         const SessionState& session_state,
         const If::Info& info);

  // Initialize by validating all the inputs, and allocating the output tensors
  Status Initialize();

  // Execute the batch, by iterating the sequence in each batch entry
  // and calling the subgraph with each item in the sequence.
  Status Execute(const FeedsFetchesManager& ffm);

 private:
  Status AllocateOutputTensors();

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const If::Info& info_;

  const std::vector<const OrtValue*>& implicit_inputs_;

  enum class AllocationType {
    Delayed,  // allocation of If output will be done by subgraph execution
    IfOutput
  };

  // track where the fetches provided to subgraph execution were allocated.
  std::vector<std::pair<AllocationType, OrtValue>> outputs_;
};

If::If(const OpKernelInfo& info) : OpKernel(info) {
  // make sure the required attributes are present even though we don't need it here.
  // The GraphProto attributes are loaded as a Graph instance by main Graph::Resolve,
  // and a SessionState instance for executing the subgraph is created by InferenceSession.
  // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
  ONNX_NAMESPACE::GraphProto proto;
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("then_branch", &proto).IsOK());
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("else_branch", &proto).IsOK());
  ORT_IGNORE_RETURN_VALUE(proto);
}

// we need this to be in the .cc so 'unique_ptr<Info> info_' can be handled
If::~If() = default;

common::Status If::SetupSubgraphExecutionInfo(const SessionState& session_state,
                                              const std::string& attribute_name,
                                              const SessionState& subgraph_session_state) {
  std::unique_ptr<If::Info>& info = attribute_name == "then_branch"
                                        ? then_info_
                                        : else_info_;

  ORT_ENFORCE(info == nullptr, "SetupSubgraphExecutionInfo should only be called once for each subgraph.");

  const auto& node = Node();
  info = std::make_unique<If::Info>(node, *subgraph_session_state.GetGraphViewer());

  // all inputs for the If subgraph are implicit
  std::vector<std::string> feed_names;
  feed_names.reserve(info->num_implicit_inputs);

  const auto& subgraph_map = subgraph_session_state.GetOrtValueNameIdxMap();

  // prune out entries that aren't in this subgraph as the 'then' and 'else' subgraphs are different
  // and implicit inputs covers both
  const auto& implicit_input_defs = node.ImplicitInputDefs();
  for (size_t i = 0, end = info->num_implicit_inputs; i < end; ++i) {
    const auto* implicit_input = implicit_input_defs[i];
    int idx;
    if (subgraph_map.GetIdx(implicit_input->Name(), idx).IsOK()) {
      feed_names.push_back(implicit_input->Name());
    } else {
      --info->num_implicit_inputs;
      info->used_implicit_inputs[i] = false;
    }
  }

  std::unique_ptr<FeedsFetchesManager> ffm;
  ORT_RETURN_IF_ERROR(FeedsFetchesManager::Create(feed_names, info->subgraph_output_names, subgraph_map, ffm));
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(subgraph_session_state, *ffm));

  // find the location all the feeds will be coming from
  std::vector<OrtDevice> feed_locations;
  controlflow::detail::FindDevicesForValues(session_state, feed_names, feed_locations);

  std::vector<const OrtMemoryInfo*> fetch_locations;
  fetch_locations.reserve(info->num_outputs);

  // we need the allocator info for each output from the If node
  // as the subgraph execution will write directly into those buffers
  const auto& outputs = node.OutputDefs();
  for (int i = 0, end = info->num_outputs; i < end; ++i) {
    // const auto& alloc_info = controlflow::detail::FindMemoryInfoForValue(session_state, outputs[i]->Name());
    const auto& alloc_info = utils::FindMemoryInfoForValue(session_state, outputs[i]->Name());
    fetch_locations.push_back(&alloc_info);
  }

  utils::FinalizeFeedFetchCopyInfo(subgraph_session_state, *ffm, feed_locations, fetch_locations);

  if (attribute_name == "then_branch")
    then_feeds_fetches_manager_ = std::move(ffm);
  else
    else_feeds_fetches_manager_ = std::move(ffm);

  return Status::OK();
}

Status If::Compute(OpKernelContext* ctx) const {
  ORT_ENFORCE(then_feeds_fetches_manager_ && else_feeds_fetches_manager_,
              "CreateFeedsFetchesManager must be called prior to execution of graph.");

  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);

  auto condition = *ctx->Input<Tensor>(0)->Data<bool>();

  auto attribute = condition ? "then_branch" : "else_branch";
  auto* session_state = ctx_internal->SubgraphSessionState(attribute);
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for '", attribute, "' attribute.");

  const auto& info = condition ? then_info_ : else_info_;
  IfImpl impl{*ctx_internal, *session_state, *info};

  auto status = impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  if (condition) {
    status = impl.Execute(*then_feeds_fetches_manager_);
  } else {
    status = impl.Execute(*else_feeds_fetches_manager_);
  }

  return status;
}

IfImpl::IfImpl(OpKernelContextInternal& context,
               const SessionState& session_state,
               const If::Info& info)
    : context_{context},
      session_state_{session_state},
      info_{info},
      implicit_inputs_{context_.GetImplicitInputs()} {
}

Status IfImpl::Initialize() {
  auto status = AllocateOutputTensors();
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

Status IfImpl::AllocateOutputTensors() {
  Status status = Status::OK();
  int index = 0;

  for (auto& graph_output : info_.subgraph.GetOutputs()) {
    auto* graph_output_shape = graph_output->Shape();
    if (!graph_output_shape) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph must have the shape set for all outputs but ",
                             graph_output->Name(), " did not.");
    }

    TensorShape output_shape = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*graph_output_shape);

    // if size < 0 we have a symbolic dimension and need to use a temporary OrtValue in the subgraph execution
    if (output_shape.Size() < 0) {
      // we still need a value to put in the feeds we give to the execution frame, so just use an empty MLValue
      outputs_.push_back({AllocationType::Delayed, {}});
    } else {
      auto* tensor = context_.Output(index, output_shape);

      if (!tensor)
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for ", graph_output->Name());

      outputs_.emplace_back(AllocationType::IfOutput, *context_.GetOutputMLValue(index));
    }

    ++index;
  }

  return Status::OK();
}

Status IfImpl::Execute(const FeedsFetchesManager& ffm) {
  Status status = Status::OK();

  // pass in implicit inputs as feeds.
  // use the FeedsFetchesInfo as that has the pruned names
  auto& feed_names = ffm.GetFeedsFetchesInfo().feed_names;

  auto num_inputs = feed_names.size();
  std::vector<OrtValue> feeds;
  feeds.reserve(num_inputs);

  // order of implicit_inputs_ matches order of feed names. skip implicit inputs that don't apply to this subgraph
  for (size_t i = 0, end = info_.used_implicit_inputs.size(); i < end; ++i) {
    if (info_.used_implicit_inputs[i]) {
      feeds.push_back(*implicit_inputs_[i]);
    }
  }

  std::vector<OrtValue> fetches;
  std::unordered_map<size_t, IExecutor::CustomAllocator> fetch_allocators;

  fetches.reserve(info_.num_outputs);
  for (int i = 0; i < info_.num_outputs; ++i) {
    fetches.push_back(outputs_[i].second);

    if (outputs_[i].first == AllocationType::Delayed) {
      // functor to forward the allocation request from the subgraph to the If node's context so that the
      // allocation plan for the If node's output is used.
      fetch_allocators[i] = [this, i](const TensorShape& shape, OrtValue& ort_value) {
        // allocate
        auto* tensor = context_.Output(i, shape);

        if (!tensor) return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for If output ", i);

        // return OrtValue for allocated tensor
        ort_value = *context_.GetOutputMLValue(i);
        return Status::OK();
      };
    }
  }

  status = utils::ExecuteSubgraph(session_state_, ffm, feeds, fetches, fetch_allocators,
                                  /*sequential_execution*/ true, context_.GetTerminateFlag(),
                                  context_.Logger());

  ORT_RETURN_IF_ERROR(status);

  return status;
}

}  // namespace onnxruntime
