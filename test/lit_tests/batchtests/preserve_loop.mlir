// RUN: enzymexlamlir-opt %s --enzyme-batch --arith-raise --canonicalize --enzyme-hlo-unroll="max-num-iterations=4" | FileCheck %s --check-prefix=PRIMAL
// RUN: enzymexlamlir-opt %s --enzyme-batch --enzyme-wrap="infn=main retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --canonicalize --enzyme-hlo-unroll="max-num-iterations=4" | FileCheck %s --check-prefix=REVERSE

// Scalar branches lowered after frontend tracing still iterate data shapes.
// Their augmented primal and reverse sweeps keep one body for a small batch.
module {
  func.func private @guarded_log(%x: tensor<f64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %positive = stablehlo.compare GT, %x, %zero : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %r = "stablehlo.if"(%positive) ({
      %log = stablehlo.log %x : tensor<f64>
      stablehlo.return %log : tensor<f64>
    }, {
      stablehlo.return %zero : tensor<f64>
    }) : (tensor<i1>) -> tensor<f64>
    return %r : tensor<f64>
  }
  func.func @main(%x: tensor<2xf64>) -> tensor<2xf64> {
    %r = enzyme.batch @guarded_log(%x) {batch_shape = array<i64: 2>} : (tensor<2xf64>) -> tensor<2xf64>
    return %r : tensor<2xf64>
  }
}

// PRIMAL-LABEL: func.func private @batched_guarded_log
// PRIMAL: stablehlo.while
// PRIMAL-SAME: enzymexla.preserve_loop
// PRIMAL: stablehlo.log
// PRIMAL-NOT: stablehlo.log
// PRIMAL: return

// REVERSE-LABEL: func.func private @diffebatched_guarded_log
// REVERSE: stablehlo.while
// REVERSE-SAME: enzymexla.preserve_loop
// REVERSE: stablehlo.if
// REVERSE: stablehlo.divide
// REVERSE-NOT: stablehlo.while
// REVERSE: return
