// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= argTys=enzyme_active,enzyme_const retTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | FileCheck %s

module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<i64>) -> tensor<f32> {
    %c0_i64 = stablehlo.constant dense<0> : tensor<i64>
    %c1_i64 = stablehlo.constant dense<1> : tensor<i64>

    %outer:2 = stablehlo.while(%iv0 = %c0_i64, %carried = %arg0) : tensor<i64>, tensor<f32>
     attributes {
      enzyme.enable_checkpointing = true,
      enzyme.checkpoint_period = 4,
      enzyme.binomial_checkpointing
    } cond {
      %cond = stablehlo.compare LT, %iv0, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %inner:2 = stablehlo.while(%iv1 = %c0_i64, %carried2 = %carried) : tensor<i64>, tensor<f32>
       cond {
        %cond2 = stablehlo.compare LT, %iv1, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %cond2 : tensor<i1>
      } do {
        %0 = stablehlo.multiply %arg0, %carried2 : tensor<f32>
        %iv1next = stablehlo.add %iv1, %c1_i64 : tensor<i64>
        stablehlo.return %iv1next, %0 : tensor<i64>, tensor<f32>
      }
      %iv0next = stablehlo.add %iv0, %c1_i64 : tensor<i64>
      stablehlo.return %iv0next, %inner#1 : tensor<i64>, tensor<f32>
    } 
    return %outer#1 : tensor<f32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<f32>, %arg1: tensor<i64>, %arg2: tensor<f32>) -> tensor<f32> {
// CHECK-NEXT:    %c = stablehlo.constant dense<0> : tensor<1xi64>
// CHECK-NEXT:    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<0xf32>
// CHECK-NEXT:    %c_1 = stablehlo.constant dense<0> : tensor<4xi64>
// CHECK-NEXT:    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK-NEXT:    %c_3 = stablehlo.constant dense<4> : tensor<i64>
// CHECK-NEXT:    %c_4 = stablehlo.constant dense<1> : tensor<i64>
// CHECK-NEXT:    %c_5 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:    %0 = stablehlo.maximum %arg1, %c_5 : tensor<i64>
// CHECK-NEXT:    %1 = stablehlo.minimum %c_3, %0 : tensor<i64>
// CHECK-NEXT:    %2:5 = stablehlo.while(%iterArg = %c_5, %iterArg_6 = %c_5, %iterArg_7 = %arg0, %iterArg_8 = %cst_2, %iterArg_9 = %c_1) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %5 = stablehlo.compare LT, %iterArg, %1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %5 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %5 = stablehlo.reshape %iterArg_7 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:      %6 = stablehlo.dynamic_update_slice %iterArg_8, %5, %iterArg : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
// CHECK-NEXT:      %7 = stablehlo.reshape %iterArg_6 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %8 = stablehlo.dynamic_update_slice %iterArg_9, %7, %iterArg : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:      %9 = stablehlo.subtract %0, %iterArg_6 : tensor<i64>
// CHECK-NEXT:      %10 = stablehlo.subtract %1, %iterArg : tensor<i64>
// CHECK-NEXT:      %11 = stablehlo.minimum %10, %9 : tensor<i64>
// CHECK-NEXT:      %12 = enzyme.binomial_progress %9, %11 : tensor<i64>
// CHECK-NEXT:      %13:2 = stablehlo.while(%iterArg_10 = %c_5, %iterArg_11 = %iterArg_7) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %16 = stablehlo.compare LT, %iterArg_10, %12 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %16 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %16:2 = stablehlo.while(%iterArg_12 = %c_5, %iterArg_13 = %iterArg_11) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %18 = stablehlo.compare LT, %iterArg_12, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %18 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %18 = stablehlo.multiply %arg0, %iterArg_13 : tensor<f32>
// CHECK-NEXT:          %19 = stablehlo.add %iterArg_12, %c_4 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %19, %18 : tensor<i64>, tensor<f32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %17 = stablehlo.add %iterArg_10, %c_4 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %17, %16#1 : tensor<i64>, tensor<f32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %14 = stablehlo.add %iterArg_6, %12 : tensor<i64>
// CHECK-NEXT:      %15 = stablehlo.add %iterArg, %c_4 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %15, %14, %13#1, %6, %8 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>
// CHECK-NEXT:    }
// CHECK-NEXT:    %3:6 = stablehlo.while(%iterArg = %c_5, %iterArg_6 = %1, %iterArg_7 = %arg2, %iterArg_8 = %2#3, %iterArg_9 = %2#4, %iterArg_10 = %cst) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>, tensor<f32>
// CHECK-NEXT:    cond {
// CHECK-NEXT:      %5 = stablehlo.compare LT, %iterArg, %0 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:      stablehlo.return %5 : tensor<i1>
// CHECK-NEXT:    } do {
// CHECK-NEXT:      %5 = stablehlo.subtract %iterArg_6, %c_4 : tensor<i64>
// CHECK-NEXT:      %6 = stablehlo.subtract %0, %iterArg : tensor<i64>
// CHECK-NEXT:      %7 = stablehlo.dynamic_slice %iterArg_8, %5, sizes = [1] : (tensor<4xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:      %8 = stablehlo.reshape %7 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:      %9 = stablehlo.dynamic_slice %iterArg_9, %5, sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %10 = stablehlo.reshape %9 : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:      %11:5 = stablehlo.while(%iterArg_11 = %10, %iterArg_12 = %5, %iterArg_13 = %8, %iterArg_14 = %iterArg_8, %iterArg_15 = %iterArg_9) : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %19 = stablehlo.add %iterArg_11, %c_4 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.compare LT, %19, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %20 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %19 = stablehlo.subtract %6, %iterArg_11 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.subtract %1, %iterArg_12 : tensor<i64>
// CHECK-NEXT:        %21 = stablehlo.minimum %20, %19 : tensor<i64>
// CHECK-NEXT:        %22 = enzyme.binomial_progress %19, %21 : tensor<i64>
// CHECK-NEXT:        %23 = stablehlo.reshape %iterArg_13 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:        %24 = stablehlo.dynamic_update_slice %iterArg_14, %23, %iterArg_12 : (tensor<4xf32>, tensor<1xf32>, tensor<i64>) -> tensor<4xf32>
// CHECK-NEXT:        %25 = stablehlo.reshape %iterArg_11 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:        %26 = stablehlo.dynamic_update_slice %iterArg_15, %25, %iterArg_12 : (tensor<4xi64>, tensor<1xi64>, tensor<i64>) -> tensor<4xi64>
// CHECK-NEXT:        %27 = stablehlo.add %iterArg_11, %22 : tensor<i64>
// CHECK-NEXT:        %28 = stablehlo.compare EQ, %27, %6 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        %29 = stablehlo.subtract %27, %c_4 : tensor<i64>
// CHECK-NEXT:        %30 = stablehlo.select %28, %29, %27 : tensor<i1>, tensor<i64>
// CHECK-NEXT:        %31:2 = stablehlo.while(%iterArg_16 = %iterArg_11, %iterArg_17 = %iterArg_13) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:        cond {
// CHECK-NEXT:          %33 = stablehlo.compare LT, %iterArg_16, %30 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:          stablehlo.return %33 : tensor<i1>
// CHECK-NEXT:        } do {
// CHECK-NEXT:          %33:2 = stablehlo.while(%iterArg_18 = %c_5, %iterArg_19 = %iterArg_17) : tensor<i64>, tensor<f32> attributes {enzymexla.checkpoint_segment}
// CHECK-NEXT:          cond {
// CHECK-NEXT:            %35 = stablehlo.compare LT, %iterArg_18, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:            stablehlo.return %35 : tensor<i1>
// CHECK-NEXT:          } do {
// CHECK-NEXT:            %35 = stablehlo.multiply %arg0, %iterArg_19 : tensor<f32>
// CHECK-NEXT:            %36 = stablehlo.add %iterArg_18, %c_4 : tensor<i64>
// CHECK-NEXT:            stablehlo.return %36, %35 : tensor<i64>, tensor<f32>
// CHECK-NEXT:          }
// CHECK-NEXT:          %34 = stablehlo.add %iterArg_16, %c_4 : tensor<i64>
// CHECK-NEXT:          stablehlo.return %34, %33#1 : tensor<i64>, tensor<f32>
// CHECK-NEXT:        }
// CHECK-NEXT:        %32 = stablehlo.add %iterArg_12, %c_4 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %27, %32, %31#1, %24, %26 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>
// CHECK-NEXT:      }
// CHECK-NEXT:      %12 = stablehlo.maximum %arg1, %c_5 : tensor<i64>
// CHECK-NEXT:      %13 = stablehlo.reshape %12 : (tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:      %14 = stablehlo.dynamic_pad %cst_0, %cst, %c, %13, %c : (tensor<0xf32>, tensor<f32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xf32>
// CHECK-NEXT:      %15:3 = stablehlo.while(%iterArg_11 = %c_5, %iterArg_12 = %11#2, %iterArg_13 = %14) : tensor<i64>, tensor<f32>, tensor<?xf32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %19 = stablehlo.compare LT, %iterArg_11, %arg1 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %19 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %19 = stablehlo.reshape %iterArg_12 : (tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:        %20 = stablehlo.dynamic_update_slice %iterArg_13, %19, %iterArg_11 : (tensor<?xf32>, tensor<1xf32>, tensor<i64>) -> tensor<?xf32>
// CHECK-NEXT:        %21 = stablehlo.multiply %arg0, %iterArg_12 : tensor<f32>
// CHECK-NEXT:        %22 = stablehlo.add %iterArg_11, %c_4 : tensor<i64>
// CHECK-NEXT:        stablehlo.return %22, %21, %20 : tensor<i64>, tensor<f32>, tensor<?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %16 = stablehlo.subtract %12, %c_4 : tensor<i64>
// CHECK-NEXT:      %17:3 = stablehlo.while(%iterArg_11 = %c_5, %iterArg_12 = %iterArg_7, %iterArg_13 = %iterArg_10) : tensor<i64>, tensor<f32>, tensor<f32>
// CHECK-NEXT:      cond {
// CHECK-NEXT:        %19 = stablehlo.compare LT, %iterArg_11, %12 : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT:        stablehlo.return %19 : tensor<i1>
// CHECK-NEXT:      } do {
// CHECK-NEXT:        %19 = stablehlo.subtract %16, %iterArg_11 : tensor<i64>
// CHECK-NEXT:        %20 = stablehlo.dynamic_slice %15#2, %19, sizes = [1] : (tensor<?xf32>, tensor<i64>) -> tensor<1xf32>
// CHECK-NEXT:        %21 = stablehlo.reshape %20 : (tensor<1xf32>) -> tensor<f32>
// CHECK-NEXT:        %22 = stablehlo.add %iterArg_11, %c_4 : tensor<i64>
// CHECK-NEXT:        %23 = stablehlo.multiply %iterArg_12, %21 : tensor<f32>
// CHECK-NEXT:        %24 = stablehlo.add %iterArg_13, %23 : tensor<f32>
// CHECK-NEXT:        %25 = stablehlo.multiply %iterArg_12, %arg0 : tensor<f32>
// CHECK-NEXT:        stablehlo.return %22, %25, %24 : tensor<i64>, tensor<f32>, tensor<f32>
// CHECK-NEXT:      }
// CHECK-NEXT:      %18 = stablehlo.add %iterArg, %c_4 : tensor<i64>
// CHECK-NEXT:      stablehlo.return %18, %11#1, %17#1, %11#3, %11#4, %17#2 : tensor<i64>, tensor<i64>, tensor<f32>, tensor<4xf32>, tensor<4xi64>, tensor<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %4 = stablehlo.add %3#5, %3#2 : tensor<f32>
// CHECK-NEXT:    return %4 : tensor<f32>
// CHECK-NEXT:  }
