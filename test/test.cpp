#include <numpp/matrix.hpp>
#include <numpp/matrix/linalg/matmul.hpp>
#include <iostream>

int main() {
    using numpp::layout;

    // ============================================================
    // BASE MATRICES
    // ============================================================

    numpp::matrix<double> ar =
        numpp::matrix<double>::full(
            256, 128, 15,
            layout::rowmajor
        );

    numpp::matrix<double> ac =
        numpp::matrix<double>::full(
            256, 128, 15,
            layout::colmajor
        );

    numpp::matrix<double> br =
        numpp::matrix<double>::full(
            128, 192, 25,
            layout::rowmajor
        );

    numpp::matrix<double> bc =
        numpp::matrix<double>::full(
            128, 192, 25,
            layout::colmajor
        );


    // ============================================================
    // 1. NORMAL CONTIGUOUS
    //
    // A = 256 x 128
    // B = 128 x 192
    // C = 256 x 192
    // ============================================================

    auto sample_1 =
        numpp::linalg::matmul(ar, br);

    auto sample_2 =
        numpp::linalg::matmul(ac, bc);


    // ============================================================
    // 2. DIFFERENT LAYOUT
    //
    // RowMajor A * ColMajor B
    // ColMajor A * RowMajor B
    // ============================================================

    auto sample_3 =
        numpp::linalg::matmul(ar, bc);

    auto sample_4 =
        numpp::linalg::matmul(ac, br);


    // ============================================================
    // 3. TRANSPOSE
    //
    // ar^T = 128 x 256
    // Untuk valid multiplication, pasangan harus 256 x N.
    // ============================================================

    auto sample_5 =
        numpp::linalg::matmul(
            numpp::transpose(ar),
            ar
        );

    auto sample_6 =
        numpp::linalg::matmul(
            numpp::transpose(ac),
            ac
        );


    // ============================================================
    // 4. SLICE BIASA, STEP = 1
    //
    // A:
    //   rows 16..143  -> 128 rows
    //   cols 0..127   -> 128 cols
    //
    // B:
    //   rows 0..127   -> 128 rows
    //   cols 16..143  -> 128 cols
    //
    // Jadi:
    // A = 128 x 128
    // B = 128 x 128
    // C = 128 x 128
    // ============================================================

    auto ar_slice =
        numpp::slice(
            ar,
            numpp::slice_range(16, 144, 1),
            numpp::slice_range(0, 128, 1)
        );

    auto br_slice =
        numpp::slice(
            br,
            numpp::slice_range(0, 128, 1),
            numpp::slice_range(16, 144, 1)
        );

    auto sample_7 =
        numpp::linalg::matmul(
            ar_slice,
            br_slice
        );


    // ============================================================
    // 5. SLICE COLMAJOR
    // ============================================================

    auto ac_slice =
        numpp::slice(
            ac,
            numpp::slice_range(16, 144, 1),
            numpp::slice_range(0, 128, 1)
        );

    auto bc_slice =
        numpp::slice(
            bc,
            numpp::slice_range(0, 128, 1),
            numpp::slice_range(16, 144, 1)
        );

    auto sample_8 =
        numpp::linalg::matmul(
            ac_slice,
            bc_slice
        );


    // ============================================================
    // 6. SLICE STEP = 2
    //
    // ar_strided:
    //   256 / 2 = 128 rows
    //   128 / 2 = 64 cols
    //
    // br_strided:
    //   128 / 2 = 64 rows
    //   192       = 192 cols
    //
    // A = 128 x 64
    // B = 64 x 192
    // C = 128 x 192
    // ============================================================

    auto ar_strided =
        numpp::slice(
            ar,
            numpp::slice_range(0, 256, 2),
            numpp::slice_range(0, 128, 2)
        );

    auto br_strided =
        numpp::slice(
            br,
            numpp::slice_range(0, 128, 2),
            numpp::slice_range(0, 192, 1)
        );

    auto sample_9 =
        numpp::linalg::matmul(
            ar_strided,
            br_strided
        );


    // ============================================================
    // 7. COLMAJOR STRIDED
    // ============================================================

    auto ac_strided =
        numpp::slice(
            ac,
            numpp::slice_range(0, 256, 2),
            numpp::slice_range(0, 128, 2)
        );

    auto bc_strided =
        numpp::slice(
            bc,
            numpp::slice_range(0, 128, 2),
            numpp::slice_range(0, 192, 1)
        );

    auto sample_10 =
        numpp::linalg::matmul(
            ac_strided,
            bc_strided
        );


    // ============================================================
    // 8. SLICE DENGAN OFFSET + STEP
    //
    // A:
    //   rows: 3..251 step 3
    //   cols: 5..124 step 3
    //
    // 83 x 40
    //
    // B:
    //   rows: 7..126 step 3
    //   cols: 4..190 step 1
    //
    // 40 x 186
    //
    // C = 83 x 186
    // ============================================================

    auto ar_hard =
        numpp::slice(
            ar,
            numpp::slice_range(3, 252, 3),
            numpp::slice_range(5, 125, 3)
        );

    auto br_hard =
        numpp::slice(
            br,
            numpp::slice_range(7, 127, 3),
            numpp::slice_range(4, 190, 1)
        );

    auto sample_11 =
        numpp::linalg::matmul(
            ar_hard,
            br_hard
        );


    // ============================================================
    // 9. STRIDED + TRANSPOSE
    //
    // ar_strided:
    //   128 x 64
    //
    // transpose:
    //   64 x 128
    //
    // dikali dengan:
    //   ar_strided = 128 x 64
    //
    // hasil:
    //   64 x 64
    // ============================================================

    auto sample_12 =
        numpp::linalg::matmul(
            numpp::transpose(ar_strided),
            ar_strided
        );


    // ============================================================
    // 10. COLMAJOR STRIDED + TRANSPOSE
    // ============================================================

    auto sample_13 =
        numpp::linalg::matmul(
            numpp::transpose(ac_strided),
            ac_strided
        );


    // ============================================================
    // 11. SLICE + TRANSPOSE + DIFFERENT LAYOUT
    //
    // A = transpose(ar_slice)
    //     128 x 128
    //
    // B = bc_slice
    //     128 x 128
    //
    // C = 128 x 128
    // ============================================================

    auto sample_14 =
        numpp::linalg::matmul(
            numpp::transpose(ar_slice),
            bc_slice
        );


    // ============================================================
    // 12. SLICE + TRANSPOSE PADA KEDUA OPERAND
    // ============================================================

    auto sample_15 =
        numpp::linalg::matmul(
            numpp::transpose(ar_slice),
            numpp::transpose(br_slice)
        );


    // ============================================================
    // PRINT
    // ============================================================

    auto print_shape = [](
        const auto& m,
        const char* name
    ) {
        std::cout
            << name
            << ": "
            << m.row()
            << "x"
            << m.col()
            << '\n';
    };

    print_shape(sample_1,  "sample_1  normal row/row");
    print_shape(sample_2,  "sample_2  normal col/col");
    print_shape(sample_3,  "sample_3  row/col");
    print_shape(sample_4,  "sample_4  col/row");

    print_shape(sample_5,  "sample_5  transpose row");
    print_shape(sample_6,  "sample_6  transpose col");

    print_shape(sample_7,  "sample_7  row slice");
    print_shape(sample_8,  "sample_8  col slice");

    print_shape(sample_9,  "sample_9  row step2");
    print_shape(sample_10, "sample_10 col step2");

    print_shape(sample_11, "sample_11 offset + step3");

    print_shape(sample_12, "sample_12 strided transpose row");
    print_shape(sample_13, "sample_13 strided transpose col");

    print_shape(sample_14, "sample_14 slice transpose mixed");
    print_shape(sample_15, "sample_15 both transpose");

    std::cout << "All sample matmul calls completed.\n";
}