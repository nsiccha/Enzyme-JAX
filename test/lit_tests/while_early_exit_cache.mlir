// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize --cse | FileCheck %s

// Reverse mode of a counted stablehlo.while with a data-dependent exit:
//
//   i = 0; t = 0; acc = 1
//   while (i < 8) & (t < 1): acc *= x; t += dt(x); i += 1
//
// Reverse mode cannot read the trip count off the condition, so it counts the
// iterations in the augmented primal and caches the count for the reverse loop.
// The cache removal used to refuse such a loop ("WhileOp does not have known
// iteration count for cache removal"). It now indexes the tape by that counter,
// sizes the tape buffers by a static bound -- here the 8 of `i < 8`, with `i`
// starting at 0 and growing by 1 -- and seeds the reverse tape index from the
// reverse loop's actual count. The second loop starts its counter from a traced
// value, which nothing can bound, so it states the bound as
// `enzymexla.max_trip_count`.

module {
  func.func private @early_exit(%x: tensor<2xf64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %one = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %tenth = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %ones = stablehlo.constant dense<1.000000e+00> : tensor<2xf64>
    %s = stablehlo.reduce(%x init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    %dt = stablehlo.multiply %s, %tenth : tensor<f64>
    %r:3 = stablehlo.while(%i = %c0, %t = %zero, %acc = %ones) : tensor<i64>, tensor<f64>, tensor<2xf64> attributes {enzyme.disable_mincut}
    cond {
      %a = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %b = stablehlo.compare LT, %t, %one : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %c = stablehlo.and %a, %b : tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %acc2 = stablehlo.multiply %acc, %x : tensor<2xf64>
      %t2 = stablehlo.add %t, %dt : tensor<f64>
      %i2 = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %i2, %t2, %acc2 : tensor<i64>, tensor<f64>, tensor<2xf64>
    }
    %sum = stablehlo.reduce(%r#2 init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    return %sum : tensor<f64>
  }

  // Same loop, but the counter starts from a traced value: only the attribute
  // can bound it.
  func.func private @early_exit_attr(%x: tensor<2xf64>, %start: tensor<i64>) -> tensor<f64> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c8 = stablehlo.constant dense<8> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %one = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %tenth = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %ones = stablehlo.constant dense<1.000000e+00> : tensor<2xf64>
    %s = stablehlo.reduce(%x init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    %dt = stablehlo.multiply %s, %tenth : tensor<f64>
    %r:3 = stablehlo.while(%i = %start, %t = %zero, %acc = %ones) : tensor<i64>, tensor<f64>, tensor<2xf64> attributes {enzyme.disable_mincut, enzymexla.max_trip_count = 8 : i64}
    cond {
      %a = stablehlo.compare LT, %i, %c8 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %b = stablehlo.compare LT, %t, %one : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %c = stablehlo.and %a, %b : tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %acc2 = stablehlo.multiply %acc, %x : tensor<2xf64>
      %t2 = stablehlo.add %t, %dt : tensor<f64>
      %i2 = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %i2, %t2, %acc2 : tensor<i64>, tensor<f64>, tensor<2xf64>
    }
    %sum = stablehlo.reduce(%r#2 init: %zero) applies stablehlo.add across dimensions = [0] : (tensor<2xf64>, tensor<f64>) -> tensor<f64>
    return %sum : tensor<f64>
  }

  func.func @main(%x: tensor<2xf64>, %start: tensor<i64>) -> (tensor<2xf64>, tensor<2xf64>) {
    %seed = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %0 = enzyme.autodiff @early_exit(%x, %seed) {activity = [#enzyme<activity enzyme_active>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<2xf64>, tensor<f64>) -> tensor<2xf64>
    %1 = enzyme.autodiff @early_exit_attr(%x, %start, %seed) {activity = [#enzyme<activity enzyme_active>, #enzyme<activity enzyme_const>], ret_activity = [#enzyme<activity enzyme_activenoneed>]} : (tensor<2xf64>, tensor<i64>, tensor<f64>) -> tensor<2xf64>
    return %0, %1 : tensor<2xf64>, tensor<2xf64>
  }
}

// The multiply's adjoint needs the loop-carried `acc`, so it is taped into a
// buffer sized by the bound and written at the iteration counter; the reverse
// loop reads it back at its own counter, which starts at the actual iteration
// count minus one. No enzyme op survives.
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
// CHECK-LABEL:   func.func private @diffeearly_exit(
// CHECK:           stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<2xf64>, {{.*}}tensor<8x2xf64>
// CHECK:             stablehlo.dynamic_update_slice %{{.*}} : (tensor<8x2xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<8x2xf64>
// CHECK:           stablehlo.subtract %{{.*}}, %c{{.*}} : tensor<i64>
// CHECK:           stablehlo.while
// CHECK:             stablehlo.dynamic_slice %{{.*}} : (tensor<8x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK:           return
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
// CHECK-LABEL:   func.func private @diffeearly_exit_attr(
// CHECK:           stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<2xf64>, {{.*}}tensor<8x2xf64>
// CHECK:             stablehlo.dynamic_update_slice %{{.*}} : (tensor<8x2xf64>, tensor<1x2xf64>, tensor<i64>, tensor<i64>) -> tensor<8x2xf64>
// CHECK:           stablehlo.while
// CHECK:             stablehlo.dynamic_slice %{{.*}} : (tensor<8x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
// CHECK:           return
// CHECK-NOT:       enzyme.{{get|set|push|pop|init}}
