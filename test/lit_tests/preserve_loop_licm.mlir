// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=elementwise_licm(0);slice_licm(0);while_licm(1)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Invariant work belongs to a required retained body even if its operands are
// defined outside: the loop may take zero trips. Unmarked loops keep LICM.
module {
  func.func @marked(%xs: tensor<2xf64>, %x: tensor<1xf64>, %n: tensor<i64>) -> tensor<1xf64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %init = stablehlo.constant dense<0.0> : tensor<1xf64>
    %loop:2 = stablehlo.while(%i = %zero, %acc = %init) : tensor<i64>, tensor<1xf64> attributes {enzymexla.preserve_loop}
    cond {
      %more = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %more : tensor<i1>
    } do {
      %slice = stablehlo.slice %xs [0:1] : (tensor<2xf64>) -> tensor<1xf64>
      %log = stablehlo.log %x : tensor<1xf64>
      %term = stablehlo.add %slice, %log : tensor<1xf64>
      %sum = stablehlo.add %acc, %term : tensor<1xf64>
      %next = stablehlo.add %i, %one : tensor<i64>
      stablehlo.return %next, %sum : tensor<i64>, tensor<1xf64>
    }
    return %loop#1 : tensor<1xf64>
  }
  func.func @unmarked(%xs: tensor<2xf64>, %x: tensor<1xf64>, %n: tensor<i64>) -> tensor<1xf64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %init = stablehlo.constant dense<0.0> : tensor<1xf64>
    %loop:2 = stablehlo.while(%i = %zero, %acc = %init) : tensor<i64>, tensor<1xf64>
    cond {
      %more = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %more : tensor<i1>
    } do {
      %slice = stablehlo.slice %xs [0:1] : (tensor<2xf64>) -> tensor<1xf64>
      %log = stablehlo.log %x : tensor<1xf64>
      %term = stablehlo.add %slice, %log : tensor<1xf64>
      %sum = stablehlo.add %acc, %term : tensor<1xf64>
      %next = stablehlo.add %i, %one : tensor<i64>
      stablehlo.return %next, %sum : tensor<i64>, tensor<1xf64>
    }
    return %loop#1 : tensor<1xf64>
  }
}

// CHECK-LABEL: func.func @marked
// CHECK-NOT: stablehlo.slice
// CHECK-NOT: stablehlo.log
// CHECK: stablehlo.while
// CHECK: do {
// CHECK: stablehlo.slice
// CHECK: stablehlo.log
// CHECK: stablehlo.return
// CHECK-LABEL: func.func @unmarked
// CHECK-DAG: stablehlo.slice
// CHECK-DAG: stablehlo.log
// CHECK: stablehlo.while
// CHECK-NOT: stablehlo.slice
// CHECK-NOT: stablehlo.log
// CHECK: return
