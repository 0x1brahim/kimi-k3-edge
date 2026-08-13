/* k3_mxfp4_quant.c - see k3_mxfp4_quant.h. The math is a faithful C99 port of
 * tools/make_tiny_checkpoint.py::mxfp4_quant (the reference the golden fixture is
 * generated with); every boundary rule below is chosen to reproduce numpy exactly:
 *
 *   - floor(log2): computed via frexp, exact for every finite f32 (see the header),
 *     so it matches numpy's float64 log2 on all inputs, not just the golden cases.
 *   - ties in the E2M1 nearest-neighbour step go to the SMALLER value, because
 *     np.argmin returns the first minimum. The exact thresholds are 0.25, 0.75,
 *     1.25, 1.75, 2.5, 3.5, 5.0 - all exactly representable in f32, and the
 *     division by 2^exp is exact, so the comparisons cannot round differently.
 *   - x < 0 test, not x <= 0: -0.0 quantises as +0.0, exactly like numpy.
 *   - scale wraps mod 256 ((exp + 127) & 0xFF), exactly like numpy's uint8 cast.
 *
 * The 255 -> 0 rule of k3_mxfp4_dequant applies at DEQUANT time; the quantiser
 * itself emits whatever (exp + 127) wraps to, mirroring the python reference.
 */
#include "k3_mxfp4_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "k3.h" /* K3_MXFP4_GROUP */

static int qfail(const char *what)
{
    fprintf(stderr, "k3_mxfp4_quant: %s\n", what);
    return -1;
}

int k3_mxfp4_quant(unsigned char *packed, unsigned char *scales, const float *w, int rows,
                   int cols)
{
    if (!packed || !scales || !w) return qfail("NULL argument");
    if (rows <= 0 || cols <= 0 || cols % K3_MXFP4_GROUP != 0)
        return qfail("shape must be positive with cols a multiple of the 32-group");

    /* The E2M1 magnitudes, ascending. The tie thresholds between neighbours are
     * the exact midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0. numpy's argmin
     * picks the FIRST minimum on a tie, i.e. the smaller magnitude. */
    static const uint8_t E2M1[8] = {0, 1, 2, 3, 4, 5, 6, 7}; /* LUT indices */

    const int g = cols / K3_MXFP4_GROUP;
    for (int r = 0; r < rows; r++) {
        const float *row = w + (size_t)r * cols;
        for (int gr = 0; gr < g; gr++) {
            const float *blk = row + (size_t)gr * K3_MXFP4_GROUP;

            /* ---- group scale ---- */
            float amax = 0.0f;
            for (int j = 0; j < K3_MXFP4_GROUP; j++) {
                const float a = fabsf(blk[j]);
                if (a > amax) amax = a;
            }
            /* numpy: exp = floor(log2(max(amax, 1e-30))) - 2, computed in float64.
             * frexp normalises amax = m * 2^e with m in [0.5, 1), so floor(log2)
             * is e - 1 for every finite amax >= 0.5... and for every amax below
             * 0.5 too (m in [0.5,1) always holds, e adjusts). amax == 0 maps to
             * the 1e-30 clamp: floor(log2(1e-30)) == -100 exactly. */
            int e;
            int expo;
            if (amax < 1e-30f) {
                expo = -102; /* floor(log2(1e-30)) - 2 = -100 - 2 */
            } else {
                frexp(amax, &e);    /* amax = m * 2^e, m in [0.5, 1) */
                expo = (e - 1) - 2; /* floor(log2 amax) - 2 */
            }
            if (expo < -127 || expo > 127) {
                fprintf(stderr,
                        "k3_mxfp4_quant: exponent %d is outside the E8M0 range "
                        "-127..127 (row %d, group %d)\n",
                        expo, r, gr);
                return -1;
            }
            scales[r * g + gr] = (unsigned char)((expo + 127) & 0xFF);

            /* ---- nibbles ---- */
            const float mult = ldexpf(1.0f, expo); /* 2^expo, exact */
            unsigned char *pk =
                packed + (size_t)r * (cols / 2) + (size_t)gr * (K3_MXFP4_GROUP / 2);
            for (int j = 0; j < K3_MXFP4_GROUP; j += 2) {
                const float x0 = blk[j] / mult, x1 = blk[j + 1] / mult;
                const float p0 = fabsf(x0), p1 = fabsf(x1);

                uint8_t n0, n1;
                if (p0 <= 0.25f)
                    n0 = E2M1[0];
                else if (p0 <= 0.75f)
                    n0 = E2M1[1];
                else if (p0 <= 1.25f)
                    n0 = E2M1[2];
                else if (p0 <= 1.75f)
                    n0 = E2M1[3];
                else if (p0 <= 2.5f)
                    n0 = E2M1[4];
                else if (p0 <= 3.5f)
                    n0 = E2M1[5];
                else if (p0 <= 5.0f)
                    n0 = E2M1[6];
                else
                    n0 = E2M1[7];

                if (p1 <= 0.25f)
                    n1 = E2M1[0];
                else if (p1 <= 0.75f)
                    n1 = E2M1[1];
                else if (p1 <= 1.25f)
                    n1 = E2M1[2];
                else if (p1 <= 1.75f)
                    n1 = E2M1[3];
                else if (p1 <= 2.5f)
                    n1 = E2M1[4];
                else if (p1 <= 3.5f)
                    n1 = E2M1[5];
                else if (p1 <= 5.0f)
                    n1 = E2M1[6];
                else
                    n1 = E2M1[7];

                if (x0 < 0.0f) n0 |= 0x8; /* -0.0 is NOT negative: no sign bit */
                if (x1 < 0.0f) n1 |= 0x8;
                pk[j / 2] = (unsigned char)(n0 | (n1 << 4)); /* low = even */
            }
        }
    }
    return 0;
}
