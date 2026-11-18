// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=gather_elementwise;elementwise_gather" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-opt | FileCheck %s

// Marked gathers keep authored indexing order; unmarked size heuristics remain.
func.func @gather_first_n64_ordinary(%x: tensor<128xf64>, %i: tensor<64x1xi64>) -> tensor<64x1xf64> {
    %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<128xf64>, tensor<64x1xi64>) -> tensor<64x1xf64>
    %s = stablehlo.sine %g : tensor<64x1xf64>
    return %s : tensor<64x1xf64>
}
// CHECK-LABEL: func.func @gather_first_n64_ordinary
// CHECK: stablehlo.gather
// CHECK-NEXT: {{.*}}stablehlo.sine
// CHECK-NEXT: return

func.func @sine_first_n64_ordinary(%x: tensor<128xf64>, %i: tensor<64x1xi64>) -> tensor<64x1xf64> {
    %s = stablehlo.sine %x : tensor<128xf64>
    %g = "stablehlo.gather"(%s, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<128xf64>, tensor<64x1xi64>) -> tensor<64x1xf64>
    return %g : tensor<64x1xf64>
}
// CHECK-LABEL: func.func @sine_first_n64_ordinary
// CHECK: stablehlo.gather
// CHECK-NEXT: {{.*}}stablehlo.sine
// CHECK-NEXT: return

func.func @gather_first_n64_retained(%x: tensor<128xf64>, %i: tensor<64x1xi64>) -> tensor<64x1xf64> {
    %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> {enzymexla.preserve_gather_order} : (tensor<128xf64>, tensor<64x1xi64>) -> tensor<64x1xf64>
    %s = stablehlo.sine %g : tensor<64x1xf64>
    return %s : tensor<64x1xf64>
}
// CHECK-LABEL: func.func @gather_first_n64_retained
// CHECK: stablehlo.gather
// CHECK-NEXT: {{.*}}stablehlo.sine
// CHECK-NEXT: return

func.func @sine_first_n64_retained(%x: tensor<128xf64>, %i: tensor<64x1xi64>) -> tensor<64x1xf64> {
    %s = stablehlo.sine %x : tensor<128xf64>
    %g = "stablehlo.gather"(%s, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> {enzymexla.preserve_gather_order} : (tensor<128xf64>, tensor<64x1xi64>) -> tensor<64x1xf64>
    return %g : tensor<64x1xf64>
}
// CHECK-LABEL: func.func @sine_first_n64_retained
// CHECK: stablehlo.sine
// CHECK-NEXT: {{.*}}stablehlo.gather
// CHECK-NEXT: return

func.func @gather_first_n128_ordinary(%x: tensor<128xf64>, %i: tensor<128x1xi64>) -> tensor<128x1xf64> {
    %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<128xf64>, tensor<128x1xi64>) -> tensor<128x1xf64>
    %s = stablehlo.sine %g : tensor<128x1xf64>
    return %s : tensor<128x1xf64>
}
// CHECK-LABEL: func.func @gather_first_n128_ordinary
// CHECK: stablehlo.gather
// CHECK-NEXT: {{.*}}stablehlo.sine
// CHECK-NEXT: return

func.func @sine_first_n128_ordinary(%x: tensor<128xf64>, %i: tensor<128x1xi64>) -> tensor<128x1xf64> {
    %s = stablehlo.sine %x : tensor<128xf64>
    %g = "stablehlo.gather"(%s, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<128xf64>, tensor<128x1xi64>) -> tensor<128x1xf64>
    return %g : tensor<128x1xf64>
}
// CHECK-LABEL: func.func @sine_first_n128_ordinary
// CHECK: stablehlo.sine
// CHECK-NEXT: {{.*}}stablehlo.gather
// CHECK-NEXT: return

func.func @gather_first_n128_retained(%x: tensor<128xf64>, %i: tensor<128x1xi64>) -> tensor<128x1xf64> {
    %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> {enzymexla.preserve_gather_order} : (tensor<128xf64>, tensor<128x1xi64>) -> tensor<128x1xf64>
    %s = stablehlo.sine %g : tensor<128x1xf64>
    return %s : tensor<128x1xf64>
}
// CHECK-LABEL: func.func @gather_first_n128_retained
// CHECK: stablehlo.gather
// CHECK-NEXT: {{.*}}stablehlo.sine
// CHECK-NEXT: return

func.func @sine_first_n128_retained(%x: tensor<128xf64>, %i: tensor<128x1xi64>) -> tensor<128x1xf64> {
    %s = stablehlo.sine %x : tensor<128xf64>
    %g = "stablehlo.gather"(%s, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> {enzymexla.preserve_gather_order} : (tensor<128xf64>, tensor<128x1xi64>) -> tensor<128x1xf64>
    return %g : tensor<128x1xf64>
}
// CHECK-LABEL: func.func @sine_first_n128_retained
// CHECK: stablehlo.sine
// CHECK-NEXT: {{.*}}stablehlo.gather
// CHECK-NEXT: return

func.func @gather_first_n256_ordinary(%x: tensor<128xf64>, %i: tensor<256x1xi64>) -> tensor<256x1xf64> {
    %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<128xf64>, tensor<256x1xi64>) -> tensor<256x1xf64>
    %s = stablehlo.sine %g : tensor<256x1xf64>
    return %s : tensor<256x1xf64>
}
// CHECK-LABEL: func.func @gather_first_n256_ordinary
// CHECK: stablehlo.sine
// CHECK-NEXT: {{.*}}stablehlo.gather
// CHECK-NEXT: return

func.func @sine_first_n256_ordinary(%x: tensor<128xf64>, %i: tensor<256x1xi64>) -> tensor<256x1xf64> {
    %s = stablehlo.sine %x : tensor<128xf64>
    %g = "stablehlo.gather"(%s, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<128xf64>, tensor<256x1xi64>) -> tensor<256x1xf64>
    return %g : tensor<256x1xf64>
}
// CHECK-LABEL: func.func @sine_first_n256_ordinary
// CHECK: stablehlo.sine
// CHECK-NEXT: {{.*}}stablehlo.gather
// CHECK-NEXT: return

func.func @gather_first_n256_retained(%x: tensor<128xf64>, %i: tensor<256x1xi64>) -> tensor<256x1xf64> {
    %g = "stablehlo.gather"(%x, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> {enzymexla.preserve_gather_order} : (tensor<128xf64>, tensor<256x1xi64>) -> tensor<256x1xf64>
    %s = stablehlo.sine %g : tensor<256x1xf64>
    return %s : tensor<256x1xf64>
}
// CHECK-LABEL: func.func @gather_first_n256_retained
// CHECK: stablehlo.gather
// CHECK-NEXT: {{.*}}stablehlo.sine
// CHECK-NEXT: return

func.func @sine_first_n256_retained(%x: tensor<128xf64>, %i: tensor<256x1xi64>) -> tensor<256x1xf64> {
    %s = stablehlo.sine %x : tensor<128xf64>
    %g = "stablehlo.gather"(%s, %i) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> {enzymexla.preserve_gather_order} : (tensor<128xf64>, tensor<256x1xi64>) -> tensor<256x1xf64>
    return %g : tensor<256x1xf64>
}
// CHECK-LABEL: func.func @sine_first_n256_retained
// CHECK: stablehlo.sine
// CHECK-NEXT: {{.*}}stablehlo.gather
// CHECK-NEXT: return
