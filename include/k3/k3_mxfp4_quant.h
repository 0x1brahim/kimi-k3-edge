/* k3_mxfp4_quant.h - quantise fp32 into OCP MX FP4 (E2M1) + E8M0 scales.
 *
 * The inverse of k3_mxfp4_dequant (k3.h). After the fix wave removed the
 * D2a GGUF cache-admit requant (the GGUF source now stores IQ1_S natively),
 * this module is exercised by the unit tests only: the requant goldens and
 * the synthetic stress in test_gguf_par/test_gguf_bind, against the python
 * golden in tests/fixtures/mxfp4_quant_golden.bin.
 *
 * SEMANTICS (byte-for-byte, mirrored from tools/make_tiny_checkpoint.py's
 * mxfp4_quant; tests/fixtures/mxfp4_quant_golden.bin holds python-generated
 * golden outputs and the C test compares bits):
 *   Per group of K3_MXFP4_GROUP=32 consecutive columns:
 *     amax   = max |w| over the group
 *     exp    = floor(log2(max(amax, 1e-30))) - 2     E2M1's top value is 6 = 1.5*2^2
 *     scale  = (exp + 127) & 0xFF                     E8M0, biased; wraps like numpy
 *     x      = w / 2^exp                              exact: the divisor is a power of 2
 *     nibble = nearest E2M1 value to |x|, ties toward the SMALLER value
 *              (numpy argmin picks the first minimum), sign bit OR-ed in when x < 0
 *   Packed layout: two elements per byte, LOW nibble = EVEN element (the engine's
 *   convention, k3.h "NIBBLE ORDER IS A CONVENTION, NOT A RULE").
 *
 * The exponent is computed with frexp rather than log2f: frexp gives floor(log2)
 * exactly (m in [0.5,1) implies floor(log2(m*2^e)) = e-1 for every finite f32), so
 * the result cannot drift from the float64 numpy reference on boundary values.
 *
 * cols must be a multiple of K3_MXFP4_GROUP; packed must hold rows*cols/2 bytes and
 * scales rows*cols/32 bytes. Returns 0 on success, -1 (with a message) on a NULL
 * pointer, a mis-sized shape, or an exponent outside E8M0's range (the reference
 * refuses the same way rather than wrapping silently).
 */
#ifndef K3_MXFP4_QUANT_H
#define K3_MXFP4_QUANT_H

#include <stddef.h>

/* Quantise one row-major [rows][cols] fp32 matrix. */
int k3_mxfp4_quant(unsigned char *packed, unsigned char *scales, const float *w, int rows,
                   int cols);

#endif /* K3_MXFP4_QUANT_H */
