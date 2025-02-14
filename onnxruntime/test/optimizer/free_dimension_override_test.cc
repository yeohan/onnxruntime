// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"
#include "core/optimizer/free_dim_override_transformer.h"
#include "core/session/inference_session.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(FreeDimensionOverrideTransformerTest, Test) {
  string model_uri = "testdata/abs_free_dimensions.onnx";

  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();

  // The model's input shape has two free dimensions, which have the denotation of DATA_BATCH
  // and DATA_CHANNEL. Supplying these overrides to the transformer should replace those free
  // dimensions with values of 1 and 42, respectively.
  std::vector<FreeDimensionOverride> overrides =
  {
    FreeDimensionOverride{ onnx::DATA_BATCH, 1 },
    FreeDimensionOverride{ onnx::DATA_CHANNEL, 42 },
  };

  auto graph_transformer = std::make_unique<FreeDimensionOverrideTransformer>(overrides);

  onnxruntime::GraphTransformerManager graph_transformation_mgr(5);
  graph_transformation_mgr.Register(std::move(graph_transformer), TransformerLevel::Level1);

  graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1);

  // Verify that the shape of the input graph has the correct values

  const auto& graph_inputs = graph.GetInputs();
  ASSERT_TRUE(graph_inputs.size() == 1); // This model only has a single input ('x')

  const auto* input_shape = graph_inputs[0]->Shape();
  ASSERT_TRUE(input_shape->dim_size() == 3); // Model takes a 3D tensor as input; two of those dimensions are (were) free dimensions

  ASSERT_TRUE(input_shape->dim(0).denotation() == onnx::DATA_BATCH);
  ASSERT_TRUE(input_shape->dim(0).has_dim_value());
  ASSERT_TRUE(input_shape->dim(0).dim_value() == 1);

  ASSERT_TRUE(input_shape->dim(1).denotation() == onnx::DATA_CHANNEL);
  ASSERT_TRUE(input_shape->dim(1).has_dim_value());
  ASSERT_TRUE(input_shape->dim(1).dim_value() == 42);
}

}  // namespace test
}  // namespace onnxruntime
