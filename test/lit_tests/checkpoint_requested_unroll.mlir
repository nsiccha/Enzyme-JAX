// RUN: enzymexlamlir-opt --enzyme-hlo-unroll %s | FileCheck %s

// CHECK-LABEL: func.func @requested
// CHECK: stablehlo.while
// CHECK: return
func.func @requested(%arg0: tensor<f64>) -> tensor<f64> {
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %limit = stablehlo.constant dense<3> : tensor<i64>
  %loop:2 = stablehlo.while(%i = %zero, %x = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.enable_checkpointing = true}
  cond {
    %pred = stablehlo.compare LT, %i, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %pred : tensor<i1>
  } do {
    %inc = stablehlo.add %i, %one : tensor<i64>

      %value = stablehlo.multiply %x, %x : tensor<f64>
      stablehlo.return %inc, %value : tensor<i64>, tensor<f64>
  }
  return %loop#1 : tensor<f64>
}

// CHECK-LABEL: func.func @disabled
// CHECK-NOT: stablehlo.while
// CHECK-COUNT-3: stablehlo.multiply
// CHECK: return
func.func @disabled(%arg0: tensor<f64>) -> tensor<f64> {
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %limit = stablehlo.constant dense<3> : tensor<i64>
  %loop:2 = stablehlo.while(%i = %zero, %x = %arg0) : tensor<i64>, tensor<f64> attributes {enzyme.enable_checkpointing = false}
  cond {
    %pred = stablehlo.compare LT, %i, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %pred : tensor<i1>
  } do {
    %inc = stablehlo.add %i, %one : tensor<i64>

      %value = stablehlo.multiply %x, %x : tensor<f64>
      stablehlo.return %inc, %value : tensor<i64>, tensor<f64>
  }
  return %loop#1 : tensor<f64>
}

// CHECK-LABEL: func.func @nested
// CHECK: stablehlo.while
// CHECK: stablehlo.while
// CHECK: return
func.func @nested(%arg0: tensor<f64>) -> tensor<f64> {
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %limit = stablehlo.constant dense<3> : tensor<i64>
  %loop:2 = stablehlo.while(%i = %zero, %x = %arg0) : tensor<i64>, tensor<f64>
  cond {
    %pred = stablehlo.compare LT, %i, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %pred : tensor<i1>
  } do {
    %inc = stablehlo.add %i, %one : tensor<i64>

      %inner:2 = stablehlo.while(%j = %zero, %y = %x) : tensor<i64>, tensor<f64> attributes {enzyme.enable_checkpointing = true}
      cond {
        %pred = stablehlo.compare LT, %j, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %pred : tensor<i1>
      } do {
        %next = stablehlo.add %j, %one : tensor<i64>
        %value = stablehlo.multiply %y, %y : tensor<f64>
        stablehlo.return %next, %value : tensor<i64>, tensor<f64>
      }
      stablehlo.return %inc, %inner#1 : tensor<i64>, tensor<f64>
  }
  return %loop#1 : tensor<f64>
}
