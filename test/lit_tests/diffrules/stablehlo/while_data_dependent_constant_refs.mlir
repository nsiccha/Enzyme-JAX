// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --cse | FileCheck %s --implicit-check-not=enzyme.push --implicit-check-not=enzyme.pop --implicit-check-not=enzyme.init --implicit-check-not="tensor<?"
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math --canonicalize --cse --drop-unsupported-attributes | stablehlo-translate --interpret

// A loop whose trip count depends on the data, with a counted loop in its
// body, inside another counted loop: the shape of a batched Jacobi eigensolve
// that stops once it has converged (a counted map over cells, a sweep loop
// that exits on convergence, a counted loop over index pairs).
//
// The reverse of the data-dependent loop replays its body, so it needs the
// values the body reads from above. Constants among them, here the counted
// loop's bounds, are rebuilt in the reverse rather than cached: inside the
// outer loop a cached copy is a per-trip tape, and the replayed counted loop
// would then start and stop at runtime values, which leaves its own caches
// without a static size or an induction variable to index them by.
//
// Each sweep squares-down y with two multiplications by a, and adds y to s:
//
//   y = y0; s = 0
//   while (y > 1e-3): (for k in 0..2: y = y * a); s = s + y
//
// At a = 0.5, y0 = 1 it runs five sweeps, so s = sum_{n=1..5} a^(2n) =
// 0.3330078125, ds/da = sum_{n=1..5} 2n a^(2n-1) = 1.76953125 and
// ds/dy0 = s. The outer loop runs it twice and adds both results, doubling
// each. All of them are exact in binary64.

module {
  func.func private @sweeps(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %r:2 = stablehlo.while(%y = %y0, %s = %zero) : tensor<f64>, tensor<f64>
    cond {
      %c = stablehlo.compare GT, %y, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %in:2 = stablehlo.while(%k = %c0, %yy = %y) : tensor<i64>, tensor<f64>
      cond {
        %c = stablehlo.compare LT, %k, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %c : tensor<i1>
      } do {
        %k2 = stablehlo.add %k, %c1 : tensor<i64>
        %y2 = stablehlo.multiply %yy, %a : tensor<f64>
        stablehlo.return %k2, %y2 : tensor<i64>, tensor<f64>
      }
      %s2 = stablehlo.add %s, %in#1 : tensor<f64>
      stablehlo.return %in#1, %s2 : tensor<f64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // The same loops inside a counted loop that runs them twice.
  func.func private @mapped(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %m:2 = stablehlo.while(%i = %c0, %t = %zero) : tensor<i64>, tensor<f64>
    cond {
      %c = stablehlo.compare LT, %i, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %r:2 = stablehlo.while(%y = %y0, %s = %zero) : tensor<f64>, tensor<f64>
      cond {
        %c = stablehlo.compare GT, %y, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
        stablehlo.return %c : tensor<i1>
      } do {
        %in:2 = stablehlo.while(%k = %c0, %yy = %y) : tensor<i64>, tensor<f64>
        cond {
          %c = stablehlo.compare LT, %k, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
          stablehlo.return %c : tensor<i1>
        } do {
          %k2 = stablehlo.add %k, %c1 : tensor<i64>
          %y2 = stablehlo.multiply %yy, %a : tensor<f64>
          stablehlo.return %k2, %y2 : tensor<i64>, tensor<f64>
        }
        %s2 = stablehlo.add %s, %in#1 : tensor<f64>
        stablehlo.return %in#1, %s2 : tensor<f64>, tensor<f64>
      }
      %t2 = stablehlo.add %t, %r#1 : tensor<f64>
      %i2 = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %i2, %t2 : tensor<i64>, tensor<f64>
    }
    return %m#1 : tensor<f64>
  }

  // The outer trips start the sweeps from different values, y0 * [1, 4][i],
  // computed in the outer loop's body: five sweeps, then six. The replay must
  // start from exactly the forward value, so the outer loop tapes it (the
  // tensor<2xf64> it carries) rather than recomputing it in the reverse pass.
  //   t = 341/1024 + 1365/1024 = 1.666015625, dt/dy0 = t,
  //   dt/da = 1.76953125 + 7.1015625 = 8.87109375.
  func.func private @mapped_scaled(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %scales = stablehlo.constant dense<[1.000000e+00, 4.000000e+00]> : tensor<2xf64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %m:2 = stablehlo.while(%i = %c0, %t = %zero) : tensor<i64>, tensor<f64>
    cond {
      %c = stablehlo.compare LT, %i, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %sl = stablehlo.dynamic_slice %scales, %i, sizes = [1] : (tensor<2xf64>, tensor<i64>) -> tensor<1xf64>
      %sc = stablehlo.reshape %sl : (tensor<1xf64>) -> tensor<f64>
      %ys = stablehlo.multiply %y0, %sc : tensor<f64>
      %r:2 = stablehlo.while(%y = %ys, %s = %zero) : tensor<f64>, tensor<f64>
      cond {
        %c = stablehlo.compare GT, %y, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
        stablehlo.return %c : tensor<i1>
      } do {
        %in:2 = stablehlo.while(%k = %c0, %yy = %y) : tensor<i64>, tensor<f64>
        cond {
          %c = stablehlo.compare LT, %k, %c2 : (tensor<i64>, tensor<i64>) -> tensor<i1>
          stablehlo.return %c : tensor<i1>
        } do {
          %k2 = stablehlo.add %k, %c1 : tensor<i64>
          %y2 = stablehlo.multiply %yy, %a : tensor<f64>
          stablehlo.return %k2, %y2 : tensor<i64>, tensor<f64>
        }
        %s2 = stablehlo.add %s, %in#1 : tensor<f64>
        stablehlo.return %in#1, %s2 : tensor<f64>, tensor<f64>
      }
      %t2 = stablehlo.add %t, %r#1 : tensor<f64>
      %i2 = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %i2, %t2 : tensor<i64>, tensor<f64>
    }
    return %m#1 : tensor<f64>
  }

  // The reverse of each sweep loop replays its body from snapshots, and the
  // replayed counted loop keeps a two-entry tape (one per trip), as at the top
  // level.
  // CHECK-LABEL: func.func private @diffesweeps
  // CHECK: enzyme.binomial_progress
  // CHECK: stablehlo.while{{.*}}tensor<2xf64>
  // CHECK-LABEL: func.func private @diffemapped
  // CHECK: enzyme.binomial_progress
  // CHECK: stablehlo.while{{.*}}tensor<2xf64>
  // The forward outer loop tapes each trip's trip count and starting value.
  // CHECK-LABEL: func.func private @diffemapped_scaled
  // CHECK: stablehlo.while{{.*}} : tensor<i64>, tensor<f64>, tensor<2xi64>, tensor<2xf64>
  // CHECK: enzyme.binomial_progress

  func.func @main() {
    %a = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %y0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %seed = stablehlo.constant dense<1.000000e+00> : tensor<f64>

    %s = stablehlo.constant dense<0.3330078125> : tensor<f64>
    %dsda = stablehlo.constant dense<1.76953125> : tensor<f64>
    %t = stablehlo.constant dense<0.666015625> : tensor<f64>
    %dtda = stablehlo.constant dense<3.5390625> : tensor<f64>

    %g:3 = enzyme.autodiff @sweeps(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    check.expect_close %g#0, %s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %g#1, %dsda, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %g#2, %s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %m:3 = enzyme.autodiff @mapped(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    check.expect_close %m#0, %t, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %m#1, %dtda, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %m#2, %t, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %t2 = stablehlo.constant dense<1.666015625> : tensor<f64>
    %dt2da = stablehlo.constant dense<8.87109375> : tensor<f64>
    %q:3 = enzyme.autodiff @mapped_scaled(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    check.expect_close %q#0, %t2, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %q#1, %dt2da, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %q#2, %t2, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    func.return
  }
}
