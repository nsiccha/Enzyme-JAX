// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --cse | FileCheck %s --implicit-check-not=enzyme.push --implicit-check-not=enzyme.pop --implicit-check-not=enzyme.init --implicit-check-not="tensor<?"
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-math --canonicalize --cse --drop-unsupported-attributes | stablehlo-translate --interpret

// Reverse mode of stablehlo.while loops whose trip count depends on the data,
// as in an adaptive ODE integrator (`t_end - t > eps && accepted < max_steps`):
// nothing in the condition counts trips, so no tape can be sized by the trip
// count. Such a loop is reversed by binomial checkpointing over the trip count
// it actually ran: the forward pass counts trips and keeps the initial carry,
// the reverse pass replays the body from snapshots (at most
// enzyme.checkpoint_period of them, 32 by default) and differentiates one step
// per iteration. No buffer is sized by a trip count. A loop whose step adjoint
// reads none of its state (@linear) needs no replay and keeps the plain reverse
// loop over the trip count.
//
// Every loop below is a geometric decay, run until the value drops to 1e-3:
//
//   y = y0; s = 0
//   while (y > 1e-3): y = y * a; s = s + y
//
// At a = 0.5, y0 = 1 it runs N = 10 trips, so s = sum_{k=1..10} a^k =
// 1023/1024, ds/da = sum_{k=1..10} k a^(k-1) = 3.9765625 and ds/dy0 = s. All of
// them are exact in binary64, so the replayed gradients must match to the ulp
// whatever the snapshot schedule.

module {
  func.func private @geometric(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%y = %y0, %s = %zero) : tensor<f64>, tensor<f64>
    cond {
      %c = stablehlo.compare GT, %y, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %y2 = stablehlo.multiply %y, %a : tensor<f64>
      %s2 = stablehlo.add %s, %y2 : tensor<f64>
      stablehlo.return %y2, %s2 : tensor<f64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // The same loop with its step in a function, as JAX emits helpers such as
  // `clip` and `_where`: the call is replayed like the ops it stands for.
  func.func private @decay_step(%y: tensor<f64>, %s: tensor<f64>, %a: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %y2 = stablehlo.multiply %y, %a : tensor<f64>
    %s2 = stablehlo.add %s, %y2 : tensor<f64>
    return %y2, %s2 : tensor<f64>, tensor<f64>
  }

  func.func private @geometric_call(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%y = %y0, %s = %zero) : tensor<f64>, tensor<f64>
    cond {
      %c = stablehlo.compare GT, %y, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %n:2 = func.call @decay_step(%y, %s, %a) : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
      stablehlo.return %n#0, %n#1 : tensor<f64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // Three snapshots for ten steps: the replay has to re-place snapshots.
  func.func private @geometric_budget3(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%y = %y0, %s = %zero) : tensor<f64>, tensor<f64> attributes {enzyme.checkpoint_period = 3 : i64}
    cond {
      %c = stablehlo.compare GT, %y, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %y2 = stablehlo.multiply %y, %a : tensor<f64>
      %s2 = stablehlo.add %s, %y2 : tensor<f64>
      stablehlo.return %y2, %s2 : tensor<f64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // Two snapshots.
  func.func private @geometric_budget2(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%y = %y0, %s = %zero) : tensor<f64>, tensor<f64> attributes {enzyme.checkpoint_period = 2 : i64}
    cond {
      %c = stablehlo.compare GT, %y, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %y2 = stablehlo.multiply %y, %a : tensor<f64>
      %s2 = stablehlo.add %s, %y2 : tensor<f64>
      stablehlo.return %y2, %s2 : tensor<f64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // A single snapshot: every step is replayed from the initial carry.
  func.func private @geometric_budget1(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%y = %y0, %s = %zero) : tensor<f64>, tensor<f64> attributes {enzyme.checkpoint_period = 1 : i64}
    cond {
      %c = stablehlo.compare GT, %y, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %y2 = stablehlo.multiply %y, %a : tensor<f64>
      %s2 = stablehlo.add %s, %y2 : tensor<f64>
      stablehlo.return %y2, %s2 : tensor<f64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // The early-exit counted shape: a counter bounds the loop at 100 trips, the
  // data ends it after 10. The buffers are still sized by the snapshot budget,
  // not by the bound.
  func.func private @early_exit(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c100 = stablehlo.constant dense<100> : tensor<i64>
    %r:3 = stablehlo.while(%i = %c0, %y = %y0, %s = %zero) : tensor<i64>, tensor<f64>, tensor<f64>
    cond {
      %b = stablehlo.compare LT, %i, %c100 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %c = stablehlo.compare GT, %y, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
      %d = stablehlo.and %b, %c : tensor<i1>
      stablehlo.return %d : tensor<i1>
    } do {
      %i2 = stablehlo.add %i, %c1 : tensor<i64>
      %y2 = stablehlo.multiply %y, %a : tensor<f64>
      %s2 = stablehlo.add %s, %y2 : tensor<f64>
      stablehlo.return %i2, %y2, %s2 : tensor<i64>, tensor<f64>, tensor<f64>
    }
    return %r#2 : tensor<f64>
  }

  // An adaptive loop inside a counted one, as in an ODE solve over several
  // output times. Each outer trip decays y to 2^-10 and rescales it by 1024,
  // so at a = 0.5 every inner loop runs ten trips:
  //   s = 3 * 1023/1024,  ds/dy0 = s,
  //   ds/da = 3 * 3.9765625 + 1023/1024 * (0 + 1 + 2) * (1024 * 10 * a^9)
  //         = 71.87109375.
  func.func private @nested_in_counted(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %scale = stablehlo.constant dense<1.024000e+03> : tensor<f64>
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %c3 = stablehlo.constant dense<3> : tensor<i64>
    %r:3 = stablehlo.while(%i = %c0, %y = %y0, %s = %zero) : tensor<i64>, tensor<f64>, tensor<f64>
    cond {
      %c = stablehlo.compare LT, %i, %c3 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %in:2 = stablehlo.while(%yy = %y, %ss = %s) : tensor<f64>, tensor<f64>
      cond {
        %c = stablehlo.compare GT, %yy, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
        stablehlo.return %c : tensor<i1>
      } do {
        %y2 = stablehlo.multiply %yy, %a : tensor<f64>
        %s2 = stablehlo.add %ss, %y2 : tensor<f64>
        stablehlo.return %y2, %s2 : tensor<f64>, tensor<f64>
      }
      %y3 = stablehlo.multiply %in#0, %scale : tensor<f64>
      %i2 = stablehlo.add %i, %c1 : tensor<i64>
      stablehlo.return %i2, %y3, %in#1 : tensor<i64>, tensor<f64>, tensor<f64>
    }
    return %r#2 : tensor<f64>
  }

  // The same, but the outer loop is adaptive too: it stops once s >= 2.5,
  // which takes the same three trips.
  func.func private @nested_in_adaptive(%a: tensor<f64>, %y0: tensor<f64>) -> tensor<f64> {
    %thr = stablehlo.constant dense<1.000000e-03> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %scale = stablehlo.constant dense<1.024000e+03> : tensor<f64>
    %stop = stablehlo.constant dense<2.500000e+00> : tensor<f64>
    %r:2 = stablehlo.while(%y = %y0, %s = %zero) : tensor<f64>, tensor<f64> attributes {enzyme.checkpoint_period = 2 : i64}
    cond {
      %c = stablehlo.compare LT, %s, %stop : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %in:2 = stablehlo.while(%yy = %y, %ss = %s) : tensor<f64>, tensor<f64> attributes {enzyme.checkpoint_period = 3 : i64}
      cond {
        %c = stablehlo.compare GT, %yy, %thr : (tensor<f64>, tensor<f64>) -> tensor<i1>
        stablehlo.return %c : tensor<i1>
      } do {
        %y2 = stablehlo.multiply %yy, %a : tensor<f64>
        %s2 = stablehlo.add %ss, %y2 : tensor<f64>
        stablehlo.return %y2, %s2 : tensor<f64>, tensor<f64>
      }
      %y3 = stablehlo.multiply %in#0, %scale : tensor<f64>
      stablehlo.return %y3, %in#1 : tensor<f64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  // s = s + a while s < 2.4. A floating-point accumulator is not counted like
  // an integer induction variable: (2.4 - s0) / a = 4.8 trips is not what
  // rounding and the comparison make of it, so the trips are counted as they
  // run. The reverse rule of an add reads no primal value, so the reverse needs
  // neither snapshots nor replays. At a = 0.5, s0 = 0 it runs N = 5 trips:
  // s = 2.5, ds/da = N = 5 and ds/ds0 = 1.
  func.func private @linear(%a: tensor<f64>, %s0: tensor<f64>) -> tensor<f64> {
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

  // An integer induction variable with a dynamic step: i += k while i < 10.
  // The plain path would count (10 - 0) / k trips, truncated: 3 at k = 3,
  // where the loop runs 4 (i = 0, 3, 6, 9). Its trips are counted as they run.
  // At a = 0.5, y0 = 1: y = a^4 = 0.0625, dy/da = 4 a^3 = 0.5, dy/dy0 = a^4.
  func.func private @strided(%a: tensor<f64>, %y0: tensor<f64>, %k: tensor<i64>) -> tensor<f64> {
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c10 = stablehlo.constant dense<10> : tensor<i64>
    %r:2 = stablehlo.while(%i = %c0, %y = %y0) : tensor<i64>, tensor<f64>
    cond {
      %c = stablehlo.compare LT, %i, %c10 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %i2 = stablehlo.add %i, %k : tensor<i64>
      %y2 = stablehlo.multiply %y, %a : tensor<f64>
      stablehlo.return %i2, %y2 : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }

  func.func @main() {
    %a = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %y0 = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %small = stablehlo.constant dense<1.000000e-04> : tensor<f64>
    %seed = stablehlo.constant dense<1.000000e+00> : tensor<f64>

    %s = stablehlo.constant dense<0.9990234375> : tensor<f64>
    %dsda = stablehlo.constant dense<3.9765625> : tensor<f64>
    %s3 = stablehlo.constant dense<2.9970703125> : tensor<f64>
    %dsda3 = stablehlo.constant dense<71.87109375> : tensor<f64>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f64>

    %g:3 = enzyme.autodiff @geometric(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    check.expect_close %g#0, %s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %g#1, %dsda, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %g#2, %s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %gc:3 = enzyme.autodiff @geometric_call(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    check.expect_close %gc#0, %s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %gc#1, %dsda, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %gc#2, %s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %b3:2 = enzyme.autodiff @geometric_budget3(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_close %b3#0, %dsda, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %b3#1, %s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %b2:2 = enzyme.autodiff @geometric_budget2(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_close %b2#0, %dsda, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %b2#1, %s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %b1:2 = enzyme.autodiff @geometric_budget1(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_close %b1#0, %dsda, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %b1#1, %s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    // y0 below the threshold: the loop never runs and s is 0 whatever a, y0.
    %z:2 = enzyme.autodiff @geometric(%a, %small, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_close %z#0, %zero, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %z#1, %zero, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %e:2 = enzyme.autodiff @early_exit(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    check.expect_close %e#0, %dsda, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %e#1, %s, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %n:3 = enzyme.autodiff @nested_in_counted(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    check.expect_close %n#0, %s3, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %n#1, %dsda3, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %n#2, %s3, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %m:3 = enzyme.autodiff @nested_in_adaptive(%a, %y0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    check.expect_close %m#0, %s3, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %m#1, %dsda3, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %m#2, %s3, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %sl = stablehlo.constant dense<2.500000e+00> : tensor<f64>
    %five = stablehlo.constant dense<5.000000e+00> : tensor<f64>
    %l:3 = enzyme.autodiff @linear(%a, %zero, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    check.expect_close %l#0, %sl, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %l#1, %five, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %l#2, %seed, max_ulp_difference = 0 : tensor<f64>, tensor<f64>

    %k3 = stablehlo.constant dense<3> : tensor<i64>
    %y4 = stablehlo.constant dense<6.250000e-02> : tensor<f64>
    %dyda4 = stablehlo.constant dense<5.000000e-01> : tensor<f64>
    %st:3 = enzyme.autodiff @strided(%a, %y0, %k3, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<f64>, tensor<i64>, tensor<f64>) -> (tensor<f64>, tensor<f64>, tensor<f64>)
    check.expect_close %st#0, %y4, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %st#1, %dyda4, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    check.expect_close %st#2, %y4, max_ulp_difference = 0 : tensor<f64>, tensor<f64>
    return
  }
}


// The forward loop runs as written plus a trip counter, and nothing is taped
// per trip. The reverse loop runs over the counted trips; it carries the
// snapshot stores (32 slots by default, slot 0 holding x0) and, per trip,
// replays from the newest snapshot, placing snapshots where
// enzyme.binomial_progress says. The step adjoint reads the replayed state.
// CHECK-LABEL: func.func private @diffegeometric(
// CHECK:         %[[FWD:[0-9]+]]:3 = stablehlo.while({{.*}}) : tensor<f64>, tensor<f64>, tensor<i64>{{$}}
// CHECK:         stablehlo.minimum %{{.+}}, %[[FWD]]#2 : tensor<i64>
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<32xf64>, tensor<32xf64>, tensor<32xi64>, tensor<f64>{{$}}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      stablehlo.compare LT, %{{.+}}, %[[FWD]]#2
// CHECK:           %[[REMAT:[0-9]+]]:7 = stablehlo.while({{.*}}) : tensor<i64>, tensor<i64>, tensor<f64>, tensor<f64>, tensor<32xf64>, tensor<32xf64>, tensor<32xi64> attributes {enzymexla.checkpoint_segment}
// CHECK:             enzyme.binomial_progress
// CHECK:             stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64> attributes {enzymexla.checkpoint_segment}
// CHECK:               stablehlo.multiply %{{.+}}, %arg0 : tensor<f64>
// CHECK:           stablehlo.multiply %{{.+}}, %[[REMAT]]#2 : tensor<f64>

// A call in the body does not keep the loop from being replayed.
// CHECK-LABEL: func.func private @diffegeometric_call(
// CHECK:         enzyme.binomial_progress

// The snapshot stores are sized by enzyme.checkpoint_period.
// CHECK-LABEL: func.func private @diffegeometric_budget3(
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xi64>, tensor<f64>{{$}}
// CHECK:           enzyme.binomial_progress

// CHECK-LABEL: func.func private @diffegeometric_budget2(
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xf64>, tensor<2xi64>, tensor<f64>{{$}}

// CHECK-LABEL: func.func private @diffegeometric_budget1(
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<1xf64>, tensor<1xf64>, tensor<1xi64>, tensor<f64>{{$}}

// Nothing is sized by the counter's bound of 100 trips. (`{{ }}return` is the
// function's return: a bare `return` would stop at the first stablehlo.return.)
// CHECK-LABEL: func.func private @diffeearly_exit(
// CHECK-NOT:     100x
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<32xi64>, tensor<32xf64>, tensor<32xf64>, tensor<32xi64>, tensor<f64>{{$}}
// CHECK-NOT:     100x
// CHECK:         {{ }}return

// The counted outer loop tapes, per outer trip, what the inner reverse starts
// from: the inner trip count and initial carry.
// CHECK-LABEL: func.func private @diffenested_in_counted(
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<3xi64>, tensor<3xf64>, tensor<3xf64>{{$}}
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<32xf64>, tensor<32xf64>, tensor<32xi64>, tensor<f64>{{$}}
// CHECK:           enzyme.binomial_progress

// Under an adaptive outer loop, the inner loop counts its trips in the outer
// step being differentiated and is reversed there with its own snapshots.
// CHECK-LABEL: func.func private @diffenested_in_adaptive(
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<2xf64>, tensor<2xf64>, tensor<2xi64>, tensor<f64>{{$}}
// CHECK:           stablehlo.while({{.*}}) : tensor<f64>, tensor<f64>, tensor<i64> attributes {enzymexla.checkpoint_segment}
// CHECK:           stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>, tensor<i64>, tensor<3xf64>, tensor<3xf64>, tensor<3xi64>, tensor<f64>{{$}}

// No snapshots and no replay: the plain reverse loop over the counted trips.
// CHECK-LABEL: func.func private @diffelinear(
// CHECK:         %[[FWD:[0-9]+]]:2 = stablehlo.while({{.*}}) : tensor<f64>, tensor<i64>{{$}}
// CHECK-NOT:     {{checkpoint_segment|tensor<[0-9]+x}}
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>{{$}}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      stablehlo.compare LT, %{{.+}}, %[[FWD]]#1
// CHECK-NOT:     {{binomial_progress|checkpoint_segment|tensor<[0-9]+x}}
// CHECK:         {{ }}return

// A dynamic step does not count the loop.
// CHECK-LABEL: func.func private @diffestrided(
// CHECK:         enzyme.binomial_progress
