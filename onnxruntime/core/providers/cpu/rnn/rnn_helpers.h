// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _WIN32
#pragma warning(disable : 4267)
#endif

#include <algorithm>
#include <functional>
#include <future>
#include <string>
#include <vector>

#include "gsl/span"
#include "gsl/gsl_algorithm"

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/framework/allocator.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

#include "core/platform/threadpool.h"

namespace onnxruntime {
class Tensor;
class OpKernelContext;

namespace rnn {
namespace detail {

enum Direction {
  kForward = 0,
  kReverse = 1,
  kBidirectional = 2
};

inline Direction MakeDirection(const std::string& direction) {
  if (direction == "forward") {
    return kForward;
  }
  if (direction == "reverse") {
    return kReverse;
  }
  if (direction == "bidirectional") {
    return kBidirectional;
  }
  ORT_THROW("Invalid 'direction' argument of '", direction,
            "'. Must be one of 'forward', 'reverse', or 'bidirectional'.");
}

/** Allocate a unique_ptr using allocator_, and return a span to the allocated memory so usage is safe
@param allocator IAllocator to use for the allocation.
@param size Allocation size. Number of elements of type TAlloc, or total size if TAlloc is 'void'.
@param unique_ptr unique_ptr that will control the lifetime of the allocated memory.
@param fill If true, fill the allocated memory with fill_value.
@param fill_value Value to use if 'fill' is true.
@returns A span to provide bounds checked access to the allocated memory.
*/
template <typename TAlloc>
gsl::span<TAlloc> Allocate(std::shared_ptr<IAllocator> allocator,
                           size_t size,
                           IAllocatorUniquePtr<TAlloc>& unique_ptr,
                           bool fill = false, TAlloc fill_value = TAlloc{}) {
  unique_ptr = IAllocator::MakeUniquePtr<TAlloc>(allocator, size);
  auto span = gsl::make_span(unique_ptr.get(), size);

  if (fill) {
    // Do't use span.begin() it will cause performance issue and stop compiler to optimize the code
    std::fill_n(unique_ptr.get(), size, fill_value);
  }

  return span;
}

// validate the common inputs to RNN, LSTM and GRU operators
Status ValidateCommonRnnInputs(const Tensor& X,
                               const Tensor& W,
                               const Tensor& R,
                               const Tensor* B,
                               int WRB_dim_1_multipler,  // multiplier used with hidden_size for W, R and B inputs
                               const Tensor* sequence_lens,
                               const Tensor* initial_h,
                               int64_t num_directions,
                               int64_t hidden_size);

/// Copy an input array repeatedly to an output array
/// @param input_begin Beginning of input
/// @param input_end End of input
/// @param output Output iterator
/// @param repetitions Number of times to repeat copy. Assumes output is sufficiently sized.
/// @returns Position of output iterator after copy is completed
template <typename TInIter, typename TOutIter>
TOutIter RepeatVectorToConstructArray(TInIter input_begin,
                                      TInIter input_end,
                                      TOutIter output,
                                      int64_t repetitions) {
  for (int64_t i = 0; i < repetitions; i++) {
    output = std::copy(input_begin, input_end, output);
  }

  return output;
}

// reverse an LSTM or GRU sequence which has shape [seq_length, batch_size, hidden_size]
// and output to shape [seq_length, num_directions, batch_size, hidden_size]
template <typename T>
void ReverseSequence(gsl::span<const T> inputs,
                     gsl::span<T> inputs_reverse,
                     gsl::span<const int> sequence_lengths,
                     const int max_sequence_length,
                     const int batch_size,
                     const int input_size,
                     const int num_directions) {
  for (int i = 0; i < batch_size; i++) {
    int seq_len = sequence_lengths[i];

#ifdef USE_OPENMP
// Parallel execute the loop.
#pragma omp parallel for
#endif
    for (int j = 0; j < seq_len; j++) {
      gsl::span<const T> src = inputs.subspan(j * batch_size * input_size + i * input_size, input_size);
      gsl::span<T> dest = inputs_reverse.subspan(num_directions * (seq_len - j - 1) * batch_size * input_size + i * input_size, input_size);

      // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
      gsl::copy(src, dest);
    }

#ifdef USE_OPENMP
// Parallel execute the loop.
#pragma omp parallel for
#endif
    for (int j = seq_len; j < max_sequence_length; j++) {
      gsl::span<const T> src = inputs.subspan(j * batch_size * input_size + i * input_size, input_size);
      gsl::span<T> dest = inputs_reverse.subspan(num_directions * j * batch_size * input_size + i * input_size, input_size);

      // Use gsl::copy instead of std::copy() to allow compiler to optimize the code
      gsl::copy(src, dest);
    }
  }
}

// A has size M x K, B has size N x K (transposed), and C has size M x N
// We check that A, B and C are large enough before calling the lower level GEMM implementation
template <typename TSpanAIter, typename TSpanBIter, typename TSpanCIter>
void ComputeGemm(const int M,
                 const int N,
                 const int K,
                 const float alpha,
                 TSpanAIter A,
                 TSpanAIter A_end,
                 const int lda,
                 TSpanBIter B,
                 TSpanBIter B_end,
                 const int ldb,
                 const float beta,
                 TSpanCIter C,
                 TSpanCIter C_end,
                 const int ldc, concurrency::ThreadPool* tp) {
  // validate all the inputs
  // need to use the lda/ldb/ldc strides which should be >= the columns for the span
  ORT_ENFORCE(lda >= K && ldb >= K && ldc >= N);
  ORT_ENFORCE(A + (M * lda - (lda - K)) <= A_end);
  ORT_ENFORCE(B + (N * ldb - (ldb - K)) <= B_end);
  ORT_ENFORCE(C + (M * ldc - (ldc - N)) <= C_end);

  ::onnxruntime::math::GemmEx<float>(
      CblasNoTrans, CblasTrans,
      M, N, K, alpha,
      &*A, lda,
      &*B, ldb, beta,
      &*C, ldc, tp);
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
const T* SafeRawConstPointer(typename gsl::span<T>::const_iterator cur,
                             typename gsl::span<T>::const_iterator end,
                             size_t size) {
  ORT_ENFORCE(cur + size <= end);
  return &*cur;
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
const T* SafeRawConstPointer(gsl::span<T> span, size_t offset, size_t size) {
  ORT_ENFORCE(offset + size <= size_t(span.size()));
  return span.data();
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
T* SafeRawPointer(typename gsl::span<T>::iterator cur,
                  typename gsl::span<T>::iterator end,
                  size_t size) {
  ORT_ENFORCE(cur + size <= end);
  return &*cur;
}

// helper to convert a span to a raw pointer
// after validating the memory covered by the span supports the size required
template <typename T>
T* SafeRawPointer(typename gsl::span<T> span, size_t offset, size_t size) {
  ORT_ENFORCE(offset + size <= size_t(span.size()));
  return span.data() + offset;
}

template <typename TLambda>
void ExecuteLambdaInParallel(const std::string& name, TLambda lambda, int max, int step,
                             onnxruntime::concurrency::ThreadPool& ttp,
                             const ::onnxruntime::logging::Logger& logger) {
  // #define NOTHREADS to execute the lambdas directly and in order if you need to do that to debug

#ifdef NOTHREADS
  ORT_UNUSED_PARAMETER(ttp);
  ORT_UNUSED_PARAMETER(logger);

  for (int i = 0; i < max; i += step) {
    (void)name;
    std::bind(lambda, i)();
  }
#else

  ORT_UNUSED_PARAMETER(name);
  ORT_UNUSED_PARAMETER(logger);

  // ORT_ENFORCE may and does throw at times from within the tasks that run
  // on a thread-pool. Without propagating exceptions the process exits silently
  // which will make diagnosing bugs more difficult.

  // \! UGLY
  // We have a problem here with the current thread-pool is that it takes std::function
  // by value and copies it more than once (even though it is movable).
  //
  // To report status and exceptions properly it's better to use
  // futures and promises but they are not copyable, so we can't come up with a functor
  // with a promise member and we are downgrading to C++11 where we can't have captures that moved in.
  //
  // At the same time promises MUST live in the child thread so if we throw from the main thread
  // we don't destroy any promises that are on the main thread stack which children threads may still be using.
  //
  // The only solution with the current Eigen that comes to mind is to have shared_ptr to with std::promise.
  //
  const int total_tasks = max / (step > 0 ? step : 1) + (max % step > 0 ? 1 : 0);
  std::vector<std::future<void> > futures;
  futures.reserve(total_tasks);

  for (int i = 0, t = 0; i < max; i += step, ++t) {
    auto p_ptr = std::make_shared<std::promise<void> >();
    futures.push_back(p_ptr->get_future());
    ttp.Schedule([p_ptr, lambda, i]() {
      try {
        lambda(i);
        p_ptr->set_value();
      } catch (...) {
        p_ptr->set_exception(std::current_exception());
      }
    });
  }

  // We'd like to wait until all of the tasks have finished
  // even though one or more have already thrown. We will store
  // the first exception and then will re-throw at the end.
  std::exception_ptr pending_exception;
  for (auto& fut : futures) {
    try {
      // get() will re-throw any exceptions
      // the running task may throw
      fut.get();
    } catch (...) {
      if (!pending_exception) {
        pending_exception = std::current_exception();
      }
    }
  }

  if (pending_exception) {
    std::rethrow_exception(pending_exception);
  }

#endif
}

void DumpMatrixImpl(const std::string& name, const float* src, int row, int col,
                    int offset = 0, int col_width = -1);

// Helper class to wrap the processing of the activation funcs and any alpha/beta values.
// The alpha/beta values are consumed in the order of the activation funcs. once they run out
// defaults will be used as needed.
// The Entries property contains the normalized function names and the alpha/beta value to use.
class ActivationFuncs {
 public:
  struct Entry {
    const std::string name;
    const float alpha;
    const float beta;
  };

  ActivationFuncs() = default;

  ActivationFuncs(const std::vector<std::string>& funcs,
                  const std::vector<float>& alphas,
                  const std::vector<float>& betas);

  const std::vector<Entry>& Entries() const {
    return entries_;
  }

 private:
  std::vector<Entry> entries_;
};

namespace deepcpu {

using AddBiasIntoFuncPtr = void (*)(const float*, float*, const int);
using ClipWithBiasFuncPtr = void (*)(float, const float*, float*, const int);
using ActivationFuncPtr = void (*)(float*, int, float, float);
using ActivationFuncBPtr = void (*)(const float*, float*, int, float, float);
using LstmMergeGatesFuncPtr = void (*)(const float*, float*, const float*, float*, int, float, float);
using GruResetGateFuncPtr = void (*)(const float*, float*, float*, int, float, float);
using GruOutputGateFuncPtr = void (*)(float*, const float*, const float*, float*, int, float, float);

ActivationFuncPtr ActivationFuncByName(const std::string& func);
LstmMergeGatesFuncPtr LstmMergeGatesFuncByName(const std::string& func);
GruResetGateFuncPtr GruResetGateFuncByName(const std::string& func);
GruOutputGateFuncPtr GruOutputGateFuncByName(const std::string& func);

void add_bias_into_ignore(const float* ignored, const float* pd, int c);
void add_bias_into(const float* ps, float* pd, int c);
void clip(float b, float* pd, int c);
void clip_add_bias(float b, const float* pb, float* pd, int c);
void clip_ignore_bias(float b, const float* pb, float* pd, int c);
void sigmoid_m(const float* ps1, float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta);
void tanh_m(const float* ps1, float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta);
void relu_m(const float* ps1, const float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta);
void sigmoid_exact_m(const float* ps1, const float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta);
void tanh_exact_m(const float* ps1, const float* ps1_c, const float* ps2, float* pd, int c, float alpha, float beta);
void sigmoid(float* pd, int c, float alpha, float beta);
void tanh(float* pd, int c, float alpha, float beta);
void relu(float* pd, int c, float alpha, float beta);
void sigmoid_exact(float* pd, int c, float alpha, float beta);
void tanh_exact(float* pd, int c, float alpha, float beta);
void merge_lstm_gates_to_memory(const float* pprev, const float* pi, const float* pf, const float* pg, float* pcurr,
                                int c);
void gru_reset_gate_tanh(const float* ps1, float* ps2, float* pd, int c, float alpha, float beta);
void gru_reset_gate_sigmoid(const float* ps1, float* ps2, float* pd, int c, float alpha, float beta);
void gru_reset_gate_relu(const float* ps1, const float* ps2, float* pd, int c, float alpha, float beta);
void gru_output_gate_tanh(float* ph, const float* pz, const float* ps, float* po, int c, float alpha, float beta);
void gru_output_gate_sigmoid(float* ph, const float* pz, const float* ps, float* po, int c, float alpha, float beta);
void gru_output_gate_relu(const float* ph, const float* pz, const float* ps, float* po, int c, float alpha, float beta);

inline void elementwise_product(const float* op1, const float* op2, float* dest, int size) {
  for (int i = 0; i < size; i++)
    dest[i] += op1[i] * op2[i];
}

inline void elementwise_sum1(const float* src, float* dest, int size) {
  for (int i = 0; i < size; i++)
    dest[i] += src[i];
}

inline void elementwise_sum2(const float* src1, const float* src2, float* dest, int size) {
  for (int i = 0; i < size; i++)
    dest[i] += src1[i] + src2[i];
}

}  // namespace deepcpu
}  // namespace detail
}  // namespace rnn
}  // namespace onnxruntime
