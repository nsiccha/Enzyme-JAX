// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=parallel_while_to_batched_scatter" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// CHECK-LABEL: func.func @requested
// CHECK: stablehlo.while
func.func @requested(%x: tensor<8x9xf64>, %y: tensor<8x9xf64>) -> tensor<8x9xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<8x9xf64> attributes {enzymexla.parallel, enzyme.enable_checkpointing = true}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 9] : (tensor<8x9xf64>, tensor<i64>, tensor<i64>) -> tensor<1x9xf64>
    %m = stablehlo.reshape %row : (tensor<1x9xf64>) -> tensor<3x3xf64>
    %t = stablehlo.transpose %m, dims = [1, 0] {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %u = stablehlo.reshape %t : (tensor<3x3xf64>) -> tensor<1x9xf64>
    %w = stablehlo.dynamic_update_slice %acc, %u, %i, %c0 : (tensor<8x9xf64>, tensor<1x9xf64>, tensor<i64>, tensor<i64>) -> tensor<8x9xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %w : tensor<i64>, tensor<8x9xf64>
  }
  return %0#1 : tensor<8x9xf64>
}


// CHECK-LABEL: func.func @disabled
// CHECK-NOT: stablehlo.while
// CHECK: stablehlo.scatter
func.func @disabled(%x: tensor<8x9xf64>, %y: tensor<8x9xf64>) -> tensor<8x9xf64> {
  %c0 = stablehlo.constant dense<0> : tensor<i64>
  %c1 = stablehlo.constant dense<1> : tensor<i64>
  %c8 = stablehlo.constant dense<8> : tensor<i64>
  %0:2 = stablehlo.while(%i = %c0, %acc = %y) : tensor<i64>, tensor<8x9xf64> attributes {enzymexla.parallel, enzyme.enable_checkpointing = false}
   cond {
    %c = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %row = stablehlo.dynamic_slice %x, %i, %c0, sizes = [1, 9] : (tensor<8x9xf64>, tensor<i64>, tensor<i64>) -> tensor<1x9xf64>
    %m = stablehlo.reshape %row : (tensor<1x9xf64>) -> tensor<3x3xf64>
    %t = stablehlo.transpose %m, dims = [1, 0] {enzymexla.symmetric_matrix = [#enzymexla.guaranteed<NOTGUARANTEED>]} : (tensor<3x3xf64>) -> tensor<3x3xf64>
    %u = stablehlo.reshape %t : (tensor<3x3xf64>) -> tensor<1x9xf64>
    %w = stablehlo.dynamic_update_slice %acc, %u, %i, %c0 : (tensor<8x9xf64>, tensor<1x9xf64>, tensor<i64>, tensor<i64>) -> tensor<8x9xf64>
    %n = stablehlo.add %i, %c1 : tensor<i64>
    stablehlo.return %n, %w : tensor<i64>, tensor<8x9xf64>
  }
  return %0#1 : tensor<8x9xf64>
}
