// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate --interpret

// The second child loop captures the first child's result. During reverse
// replay, that capture must refer to this replay's value, not a sibling
// region in the original augmented primal. A later sibling also reads the
// capture, checking that a child's popped-cache mapping does not escape.
module {
  func.func private @nested(%x: tensor<f64>, %n: tensor<i64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %three = stablehlo.constant dense<3> : tensor<i64>
    %fzero = stablehlo.constant dense<0.0> : tensor<f64>
    %fone = stablehlo.constant dense<1.0> : tensor<f64>
    %outer:2 = stablehlo.while(%i = %zero, %acc = %fzero) : tensor<i64>, tensor<f64>
      attributes {enzyme.enable_checkpointing = true, enzyme.checkpoint_period = 4 : i64, enzyme.binomial_checkpointing}
      cond {
        %pred = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %pred : tensor<i1>
      } do {
        %first:2 = stablehlo.while(%j = %zero, %a = %x) : tensor<i64>, tensor<f64>
          attributes {enzyme.enable_checkpointing = true, enzyme.checkpoint_period = 4 : i64, enzyme.binomial_checkpointing}
          cond {
            %pred = stablehlo.compare LT, %j, %three : (tensor<i64>, tensor<i64>) -> tensor<i1>
            stablehlo.return %pred : tensor<i1>
          } do {
            %a_next = stablehlo.multiply %a, %x : tensor<f64>
            %j_next = stablehlo.add %j, %one : tensor<i64>
            stablehlo.return %j_next, %a_next : tensor<i64>, tensor<f64>
          }
        %second:2 = stablehlo.while(%k = %zero, %b = %fone) : tensor<i64>, tensor<f64>
          attributes {enzyme.enable_checkpointing = true, enzyme.checkpoint_period = 4 : i64, enzyme.binomial_checkpointing}
          cond {
            %pred = stablehlo.compare LT, %k, %three : (tensor<i64>, tensor<i64>) -> tensor<i1>
            stablehlo.return %pred : tensor<i1>
          } do {
            %b_next = stablehlo.multiply %b, %first#1 : tensor<f64>
            %k_next = stablehlo.add %k, %one : tensor<i64>
            stablehlo.return %k_next, %b_next : tensor<i64>, tensor<f64>
          }
        %term = stablehlo.add %second#1, %first#1 : tensor<f64>
        %acc_next = stablehlo.add %acc, %term : tensor<f64>
        %i_next = stablehlo.add %i, %one : tensor<i64>
        stablehlo.return %i_next, %acc_next : tensor<i64>, tensor<f64>
      }
    return %outer#1 : tensor<f64>
  }
  func.func @main() {
    %x = stablehlo.constant dense<1.0> : tensor<f64>
    %n = stablehlo.constant dense<6> : tensor<i64>
    %seed = stablehlo.constant dense<1.0> : tensor<f64>
    %expected_value = stablehlo.constant dense<12.0> : tensor<f64>
    %expected_gradient = stablehlo.constant dense<96.0> : tensor<f64>
    %result:2 = enzyme.autodiff @nested(%x, %n, %seed) {
      activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>],
      ret_activity = [#enzyme.activity<enzyme_active>]
    } : (tensor<f64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_almost_eq %result#0, %expected_value : tensor<f64>
    check.expect_almost_eq %result#1, %expected_gradient : tensor<f64>
    return
  }
}

// CHECK-LABEL: func.func private @diffenested
// CHECK: stablehlo.while
// CHECK-NOT: enzyme.push
// CHECK-NOT: enzyme.pop
