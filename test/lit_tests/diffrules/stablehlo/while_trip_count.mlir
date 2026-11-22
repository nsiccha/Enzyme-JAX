// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math --canonicalize --cse --drop-unsupported-attributes | stablehlo-translate --interpret

// The reverse of each loop must run as many trips as its forward did. Each
// loop adds `a` to an accumulator once per trip, so d(result)/da is that trip
// count, and the checks compare it exactly.

module {
  // A floating-point induction variable: s = 0, 0.5, ..., 2.0 passes the test,
  // so the loop runs 5 trips, where (2.4 - 0) / 0.5 = 4.8 truncates to 4.
  func.func private @float_iv(%a: tensor<f64>, %s0: tensor<f64>) -> tensor<f64> {
    %stop = stablehlo.constant dense<2.400000e+00> : tensor<f64>
    %r = stablehlo.while(%s = %s0) : tensor<f64>
    cond {
      %c = stablehlo.compare LT, %s, %stop : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %s2 = stablehlo.add %s, %a : tensor<f64>
      stablehlo.return %s2 : tensor<f64>
    }
    return %r : tensor<f64>
  }

  // Rounding: 0.1 added ten times is 0.9999999999999999 < 1.0, so this loop
  // runs 11 trips although (1.0 - 0) / 0.1 is 10 exactly. Rounding the
  // division up would not help; no closed form counts a floating-point loop.
  func.func private @float_iv_rounding(%a: tensor<f64>, %s0: tensor<f64>) -> tensor<f64> {
    %stop = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %r = stablehlo.while(%s = %s0) : tensor<f64>
    cond {
      %c = stablehlo.compare LT, %s, %stop : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %s2 = stablehlo.add %s, %a : tensor<f64>
      stablehlo.return %s2 : tensor<f64>
    }
    return %r : tensor<f64>
  }

  // An integer induction variable stepping by 2 to a runtime limit: for n = 5,
  // i = 0, 2, 4 is 3 trips, where (5 - 0) / 2 truncates to 2. For n = -3, the
  // limit is below the start and the loop runs no trip.
  func.func private @int_iv_step2(%a: tensor<f64>, %n: tensor<i64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c2 = stablehlo.constant dense<2> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%i = %c0, %x = %zero) : tensor<i64>, tensor<f64>
    cond {
      %c = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %i2 = stablehlo.add %i, %c2 : tensor<i64>
      %x2 = stablehlo.add %x, %a : tensor<f64>
      stablehlo.return %i2, %x2 : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // A runtime step as well: i = 1, 4, 7 while i < 8 with step 3 is 3 trips,
  // where (8 - 1) / 3 truncates to 2.
  func.func private @int_iv_dynamic_step(%a: tensor<f64>, %n: tensor<i64>, %k: tensor<i64>) -> tensor<f64> {
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%i = %c1, %x = %zero) : tensor<i64>, tensor<f64>
    cond {
      %c = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %i2 = stablehlo.add %i, %k : tensor<i64>
      %x2 = stablehlo.add %x, %a : tensor<f64>
      stablehlo.return %i2, %x2 : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // A loop that runs no trip, with a negative constant step: the ceiling
  // division (limit - start + step - 1) / step gives (3 - 3 - 1 - 1) / -1 = 2.
  func.func private @int_iv_empty_negative_step(%a: tensor<f64>) -> tensor<f64> {
    %c3 = stablehlo.constant dense<3> : tensor<i64>
    %cm1 = stablehlo.constant dense<-1> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%i = %c3, %x = %zero) : tensor<i64>, tensor<f64>
    cond {
      %c = stablehlo.compare LT, %i, %c3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %i2 = stablehlo.add %i, %cm1 : tensor<i64>
      %x2 = stablehlo.add %x, %a : tensor<f64>
      stablehlo.return %i2, %x2 : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // The same loop with a body that leaves the induction variable alone, a
  // step of 0: that division is then by zero.
  func.func private @int_iv_empty_zero_step(%a: tensor<f64>) -> tensor<f64> {
    %c3 = stablehlo.constant dense<3> : tensor<i64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%i = %c3, %x = %zero) : tensor<i64>, tensor<f64>
    cond {
      %c = stablehlo.compare LT, %i, %c3, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %x2 = stablehlo.add %x, %a : tensor<f64>
      stablehlo.return %i, %x2 : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  func.func @main() {
    %seed = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>

    %half = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %f:3 = enzyme.autodiff @float_iv(%half, %zero, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    %f_s = stablehlo.constant dense<2.500000e+00> : tensor<f64>
    %f_da = stablehlo.constant dense<5.000000e+00> : tensor<f64>
    check.expect_close %f#0, %f_s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %f#1, %f_da, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %f#2, %seed, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %tenth = stablehlo.constant dense<1.000000e-01> : tensor<f64>
    %g:3 = enzyme.autodiff @float_iv_rounding(%tenth, %zero, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    %g_s = stablehlo.constant dense<1.0999999999999999> : tensor<f64>
    %g_da = stablehlo.constant dense<1.100000e+01> : tensor<f64>
    check.expect_close %g#0, %g_s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %g#1, %g_da, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %g#2, %seed, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %three_halves = stablehlo.constant dense<1.500000e+00> : tensor<f64>
    %three = stablehlo.constant dense<3.000000e+00> : tensor<f64>

    %n5 = stablehlo.constant dense<5> : tensor<i64>
    %h:2 = enzyme.autodiff @int_iv_step2(%half, %n5, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_close %h#0, %three_halves, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %h#1, %three, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %nneg = stablehlo.constant dense<-3> : tensor<i64>
    %e:2 = enzyme.autodiff @int_iv_step2(%half, %nneg, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_close %e#0, %zero, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %e#1, %zero, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %n8 = stablehlo.constant dense<8> : tensor<i64>
    %k3 = stablehlo.constant dense<3> : tensor<i64>
    %d:2 = enzyme.autodiff @int_iv_dynamic_step(%half, %n8, %k3, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<i64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_close %d#0, %three_halves, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %d#1, %three, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %p:2 = enzyme.autodiff @int_iv_empty_negative_step(%half, %seed) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_close %p#0, %zero, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %p#1, %zero, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %q:2 = enzyme.autodiff @int_iv_empty_zero_step(%half, %seed) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_close %q#0, %zero, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %q#1, %zero, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    return
  }
}
