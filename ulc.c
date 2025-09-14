/**************************************/
//! ulc-codec: Ultra-Low-Complexity Audio Codec
//! Copyright (C) 2022, Ruben Nunez (Aikku; aik AT aol DOT com DOT au)
//! Refer to the project README file for license terms.
/**************************************/
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <libdragon.h>

/**************************************/
#include "ulc.h"


#define FOURIER_ASSUME(Cond)
#define FOURIER_ASSUME_ALIGNED(x,Align)
#define FOURIER_FORCE_INLINE static inline

#if (defined(FOURIER_IS_X86) && defined(__AVX__) && defined(FOURIER_ALLOW_AVX))
	typedef __m256 Fourier_Vec_t;
	#define FOURIER_VSTRIDE            8
	#define FOURIER_ALIGNMENT          32
	#define FOURIER_VLOAD(Src)         _mm256_load_ps(Src)
	#define FOURIER_VLOADU(Src)        _mm256_loadu_ps(Src)
	#define FOURIER_VSTORE(Dst, x)     _mm256_store_ps(Dst, x)
	#define FOURIER_VSTOREU(Dst, x)    _mm256_storeu_ps(Dst, x)
	#define FOURIER_VSET1(x)           _mm256_set1_ps(x)
	#define FOURIER_VSET_LINEAR_RAMP() _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f)
	#define FOURIER_VADD(x, y)         _mm256_add_ps(x, y)
	#define FOURIER_VSUB(x, y)         _mm256_sub_ps(x, y)
	#define FOURIER_VMUL(x, y)         _mm256_mul_ps(x, y)
	#define FOURIER_VREVERSE_LANE(x)   _mm256_shuffle_ps(x, x, 0x1B)
	#define FOURIER_VREVERSE(x)        _mm256_permute2f128_ps(FOURIER_VREVERSE_LANE(x), FOURIER_VREVERSE_LANE(x), 0x01)
	#define FOURIER_VNEGATE_ODD(x)     _mm256_xor_ps(x, _mm256_setr_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f))
	#if (defined(__FMA__) && defined(FOURIER_ALLOW_FMA))
		#define FOURIER_VFMA(x, y, a)     _mm256_fmadd_ps(x, y, a)
		#define FOURIER_VFMS(x, y, a)     _mm256_fmsub_ps(x, y, a)
		#define FOURIER_VNFMA(x, y, a)    _mm256_fnmadd_ps(x, y, a)
	#else
		#define FOURIER_VFMA(x, y, a)     _mm256_add_ps(_mm256_mul_ps(x, y), a)
		#define FOURIER_VFMS(x, y, a)     _mm256_sub_ps(_mm256_mul_ps(x, y), a)
		#define FOURIER_VNFMA(x, y, a)    _mm256_sub_ps(a, _mm256_mul_ps(x, y))
	#endif
#else
	typedef float Fourier_Vec_t;
	#define FOURIER_VSTRIDE			1
	#define FOURIER_ALIGNMENT		  4
	#define FOURIER_VLOAD(Src)		 (*Src)
	#define FOURIER_VLOADU(Src)		(*Src)
	#define FOURIER_VSTORE(Dst, x)	 (*Dst = x)
	#define FOURIER_VSTOREU(Dst, x)	(*Dst = x)
	#define FOURIER_VSET1(x)		   (x)
	#define FOURIER_VSET_LINEAR_RAMP() (0.0f)
	#define FOURIER_VADD(x, y)		 ((x) + (y))
	#define FOURIER_VSUB(x, y)		 ((x) - (y))
	#define FOURIER_VMUL(x, y)		 ((x) * (y))
	#define FOURIER_VREVERSE(x)		(x)
	#define FOURIER_VNEGATE_ODD(x)	 (x)
	#define FOURIER_VFMA(x, y, a)	  ((x)*(y) + (a))
	#define FOURIER_VFMS(x, y, a)	  ((x)*(y) - (a))
	#define FOURIER_VNFMA(x, y, a)	 ((a) - (x)*(y))
	#define FOURIER_VSQRT(x)		   fm_sqrtf(x)
#endif

// #define Fourier_Sin(x) fm_sinf(x)
// #define Fourier_Cos(x) fm_cosf(x)
FOURIER_FORCE_INLINE Fourier_Vec_t Fourier_Sin(Fourier_Vec_t x) {
	Fourier_Vec_t x2  = FOURIER_VMUL(x, x);
	Fourier_Vec_t Res = FOURIER_VFMA(x2, FOURIER_VSET1(+0x1.3C8B08p-13f), FOURIER_VSET1(-0x1.3237ECp-8f));
				  Res = FOURIER_VFMA(x2, Res, FOURIER_VSET1(+0x1.4667ACp-4f));
				  Res = FOURIER_VFMA(x2, Res, FOURIER_VSET1(-0x1.4ABBB8p-1f));
				  Res = FOURIER_VFMA(x2, Res, FOURIER_VSET1(+0x1.921FB4p0f));
	return FOURIER_VMUL(x, Res);
}

/************************************************/

// Cos[z] approximation (where z = x*Pi/2, x < Pi/2)
// Coefficients derived by RMSE minimization
FOURIER_FORCE_INLINE Fourier_Vec_t Fourier_Cos(Fourier_Vec_t x) {
	Fourier_Vec_t x2  = FOURIER_VMUL(x, x);
	Fourier_Vec_t Res = FOURIER_VFMA(x2, FOURIER_VSET1(+0x1.C2EA62p-11f), FOURIER_VSET1(-0x1.550810p-6f));
				  Res = FOURIER_VFMA(x2, Res, FOURIER_VSET1(+0x1.03BDD6p-2f));
				  Res = FOURIER_VFMA(x2, Res, FOURIER_VSET1(-0x1.3BD3B2p0f));
				  Res = FOURIER_VFMA(x2, Res, FOURIER_VSET1(1.0f));
	return Res;
}

FOURIER_FORCE_INLINE void Fourier_DCT2(float *Buf, float *Tmp, int N); 
FOURIER_FORCE_INLINE void Fourier_DCT2_8(float *x);
FOURIER_FORCE_INLINE void Fourier_DCT4(float *Buf, float *Tmp, int N);
FOURIER_FORCE_INLINE void Fourier_DCT4_8(float *x);

FOURIER_FORCE_INLINE void Fourier_DCT2_8(float *x) {
	const float sqrt1_2 = 0x1.6A09E6p-1f;
	const float c1_4 = 0x1.F6297Cp-1f, s1_4 = 0x1.8F8B84p-3f;
	const float c3_4 = 0x1.A9B662p-1f, s3_4 = 0x1.1C73B4p-1f;
	const float c6_4 = 0x1.87DE2Ap-2f, s6_4 = 0x1.D906BCp-1f;

	// First stage butterflies (DCT2_8)
	float s07 = x[0]+x[7];
	float d07 = x[0]-x[7];
	float s16 = x[1]+x[6];
	float d16 = x[1]-x[6];
	float s25 = x[2]+x[5];
	float d25 = x[2]-x[5];
	float s34 = x[3]+x[4];
	float d34 = x[3]-x[4];

	// Second stage (DCT2_4, DCT4_4)
	float ss07s34 = s07+s34;
	float ds07s34 = s07-s34;
	float ss16s25 = s16+s25;
	float ds16s25 = s16-s25;
	float d34d07x =  c3_4*d34 + s3_4*d07;
	float d34d07y = -s3_4*d34 + c3_4*d07;
	float d25d16x =  c1_4*d25 + s1_4*d16;
	float d25d16y = -s1_4*d25 + c1_4*d16;

	// Third stage (rotation butterflies; DCT2_2, DCT4_2, DCT2_2, DCT2_2)
	float a0 =	   ss07s34 +	  ss16s25;
	float b0 =	   ss07s34 -	  ss16s25;
	float c0 =  c6_4*ds16s25 + s6_4*ds07s34;
	float d0 = -s6_4*ds16s25 + c6_4*ds07s34;
	float a1 =	   d34d07y +	  d25d16x;
	float c1 =	   d34d07y -	  d25d16x;
	float d1 =	   d34d07x +	  d25d16y;
	float b1 =	   d34d07x -	  d25d16y;

	// Permute and final DCT4 stage
	x[0] = a0;
	x[4] = b0 * sqrt1_2;
	x[2] = c0;
	x[6] = d0;
	x[1] = (a1 + d1) * sqrt1_2;
	x[5] = b1;
	x[3] = c1;
	x[7] = (a1 - d1) * sqrt1_2;
}


FOURIER_FORCE_INLINE void Fourier_DCT2(float *Buf, float *Tmp, int N) {
	int i;
	FOURIER_ASSUME_ALIGNED(Buf, FOURIER_ALIGNMENT);
	FOURIER_ASSUME_ALIGNED(Tmp, FOURIER_ALIGNMENT);
	FOURIER_ASSUME(N >= 8);

	// Stop condition
	if(N == 8) {
		Fourier_DCT2_8(Buf);
		return;
	}

	// Perform butterflies
	//  u = H_n.x
	{
		const float *SrcLo = Buf;
		const float *SrcHi = Buf + N;
			  float *DstLo = Tmp;
			  float *DstHi = Tmp + N/2;
#if FOURIER_VSTRIDE > 1
		Fourier_Vec_t a, b;
		Fourier_Vec_t s, d;
		for(i=0;i<N/2;i+=FOURIER_VSTRIDE) {
			SrcHi -= FOURIER_VSTRIDE; b = FOURIER_VREVERSE(FOURIER_VLOAD(SrcHi));
			a = FOURIER_VLOAD(SrcLo); SrcLo += FOURIER_VSTRIDE;
			s = FOURIER_VADD(a, b);
			d = FOURIER_VSUB(a, b);
			FOURIER_VSTORE(DstLo, s); DstLo += FOURIER_VSTRIDE;
			FOURIER_VSTORE(DstHi, d); DstHi += FOURIER_VSTRIDE;
		}
#else
		float a, b;
		for(i=0;i<N/2;i++) {
			a = *SrcLo++;
			b = *--SrcHi;
			*DstLo++ = a + b;
			*DstHi++ = a - b;
		}
#endif
	}

	// Perform recursion
	//  z1 = cos2([u_j][j=0..n/2-1],n/2)
	//  z2 = cos4([u_j][j=n/2..n-1],n/2)
	Fourier_DCT2(Tmp,	   Buf,	   N/2);
	Fourier_DCT4(Tmp + N/2, Buf + N/2, N/2);

	// Combine
	//  y = (P_n)^T.(z1^T, z2^T)^T
	{
		const float *SrcLo = Tmp;
		const float *SrcHi = Tmp + N/2;
			  float *Dst   = Buf;
#if FOURIER_VSTRIDE > 1
		Fourier_Vec_t a, b;
		for(i=0;i<N/2;i+=FOURIER_VSTRIDE) {
			a = FOURIER_VLOAD(SrcLo); SrcLo += FOURIER_VSTRIDE;
			b = FOURIER_VLOAD(SrcHi); SrcHi += FOURIER_VSTRIDE;
			FOURIER_VINTERLEAVE(a, b, &a, &b);
			FOURIER_VSTORE(Dst+0,			   a);
			FOURIER_VSTORE(Dst+FOURIER_VSTRIDE, b); Dst += 2*FOURIER_VSTRIDE;
		}
#else
		for(i=0;i<N/2;i++) {
			*Dst++ = *SrcLo++;
			*Dst++ = *SrcHi++;
		}
#endif
	}
}

FOURIER_FORCE_INLINE void Fourier_DCT4_8(float *x) {
	const float sqrt1_2 = 0x1.6A09E6p-1f;
	const float c1_3 = 0x1.D906BCp-1f, s1_3 = 0x1.87DE2Ap-2f;
	const float c1_5 = 0x1.FD88DAp-1f, s1_5 = 0x1.917A6Cp-4f;
	const float c3_5 = 0x1.E9F416p-1f, s3_5 = 0x1.294062p-2f;
	const float c5_5 = 0x1.C38B30p-1f, s5_5 = 0x1.E2B5D4p-2f;
	const float c7_5 = 0x1.8BC806p-1f, s7_5 = 0x1.44CF32p-1f;

	// First stage (rotation butterflies; DCT4_8)
	float ax =  c1_5*x[0] + s1_5*x[7];
	float ay =  s1_5*x[0] - c1_5*x[7];
	float bx =  c3_5*x[1] + s3_5*x[6];
	float by = -s3_5*x[1] + c3_5*x[6];
	float cx =  c5_5*x[2] + s5_5*x[5];
	float cy =  s5_5*x[2] - c5_5*x[5];
	float dx =  c7_5*x[3] + s7_5*x[4];
	float dy = -s7_5*x[3] + c7_5*x[4];

	// Second stage (butterflies; DCT2_4)
	float saxdx = ax + dx;
	float daxdx = ax - dx;
	float sbxcx = bx + cx;
	float dbxcx = bx - cx;
	float sdyay = dy + ay;
	float ddyay = dy - ay;
	float scyby = cy + by;
	float dcyby = cy - by;

	// Third stage (rotation butterflies; DCT2_2, DCT4_2)
	float sx =	  saxdx +	  sbxcx;
	float sy =	  saxdx -	  sbxcx;
	float tx = c1_3*daxdx + s1_3*dbxcx;
	float ty = s1_3*daxdx - c1_3*dbxcx;
	float ux =	  sdyay +	  scyby;
	float uy =	  sdyay -	  scyby;
	float vx = c1_3*ddyay + s1_3*dcyby;
	float vy = s1_3*ddyay - c1_3*dcyby;

	// Permute and final DCT4 stage
	x[0] = sx;
	x[1] = (tx - vy);
	x[2] = (tx + vy);
	x[3] = (sy + uy) * sqrt1_2;
	x[4] = (sy - uy) * sqrt1_2;
	x[5] = (ty - vx);
	x[6] = (ty + vx);
	x[7] = ux;
}


FOURIER_FORCE_INLINE void Fourier_DCT4(float *Buf, float *Tmp, const int N) {
	int i;
	FOURIER_ASSUME_ALIGNED(Buf, FOURIER_ALIGNMENT);
	FOURIER_ASSUME_ALIGNED(Tmp, FOURIER_ALIGNMENT);
	FOURIER_ASSUME(N >= 8);

	// Stop condition
	if(N == 8) {
		Fourier_DCT4_8(Buf);
		return;
	}

	// Perform rotation butterflies
	//  u = R_n.x
	{
		const float *SrcLo = Buf;
		const float *SrcHi = Buf + N;
			  float *DstLo = Tmp;
			  float *DstHi = Tmp + N/2;
#if FOURIER_VSTRIDE > 1
		Fourier_Vec_t a, b;
		Fourier_Vec_t t0, t1 = FOURIER_VMUL(FOURIER_VSET1(1.0f/N), FOURIER_VADD(FOURIER_VSET_LINEAR_RAMP(), FOURIER_VSET1(0.5f)));
		Fourier_Vec_t c  = Fourier_Cos(t1);
		Fourier_Vec_t s  = Fourier_Sin(t1);
		Fourier_Vec_t wc = Fourier_Cos(FOURIER_VSET1((float)FOURIER_VSTRIDE / N));
		Fourier_Vec_t ws = Fourier_Sin(FOURIER_VSET1((float)FOURIER_VSTRIDE / N));
		for(i=0;i<N/2;i+=FOURIER_VSTRIDE) {
			SrcHi -= FOURIER_VSTRIDE; b = FOURIER_VREVERSE(FOURIER_VLOAD(SrcHi));
			a = FOURIER_VLOAD(SrcLo); SrcLo += FOURIER_VSTRIDE;
			t1 = FOURIER_VMUL(s, a);
			t0 = FOURIER_VMUL(c, a);
			t1 = FOURIER_VNFMA(c, b, t1);
			t0 = FOURIER_VFMA (s, b, t0);
			t1 = FOURIER_VNEGATE_ODD(t1);
			FOURIER_VSTORE(DstLo, t0); DstLo += FOURIER_VSTRIDE;
			FOURIER_VSTORE(DstHi, t1); DstHi += FOURIER_VSTRIDE;
			t0 = c;
			t1 = s;
			c = FOURIER_VNFMA(t1, ws, FOURIER_VMUL(t0, wc));
			s = FOURIER_VFMA (t1, wc, FOURIER_VMUL(t0, ws));
		}
#else
		float a, b;
		float c  = Fourier_Cos(0.5f / N);
		float s  = Fourier_Sin(0.5f / N);
		float wc = Fourier_Cos(1.0f / N);
		float ws = Fourier_Sin(1.0f / N);
		for(i=0;i<N/2;i+=2) {
			a = *SrcLo++;
			b = *--SrcHi;
			*DstLo++ =  c*a + s*b;
			*DstHi++ =  s*a - c*b;
			a = c;
			b = s;
			c = wc*a - ws*b;
			s = ws*a + wc*b;

			a = *SrcLo++;
			b = *--SrcHi;
			*DstLo++ =  c*a + s*b;
			*DstHi++ = -s*a + c*b; // <- Sign-flip for DST
			a = c;
			b = s;
			c = wc*a - ws*b;
			s = ws*a + wc*b;
		}
#endif
	}

	// Perform recursion
	//  z1 = cos2([u_j][j=0..n/2-1],n/2)
	//  z2 = cos2([u_j][j=n/2..n-1],n/2)
	Fourier_DCT2(Tmp,	   Buf,	   N/2);
	Fourier_DCT2(Tmp + N/2, Buf + N/2, N/2);

	// Combine
	//  w = U_n.(z1^T, z2^T)^T
	//  y = (P_n)^T.w
	{
		const float *TmpLo = Tmp;
		const float *TmpHi = Tmp + N;
			  float *Dst   = Buf;
		*Dst++ = *TmpLo++;
#if FOURIER_VSTRIDE > 1
		{
			Fourier_Vec_t a, b;
			Fourier_Vec_t t0, t1;
			for(i=0;i<N/2-FOURIER_VSTRIDE;i+=FOURIER_VSTRIDE) {
				TmpHi -= FOURIER_VSTRIDE; b = FOURIER_VREVERSE(FOURIER_VLOAD(TmpHi));
				a = FOURIER_VLOADU(TmpLo); TmpLo += FOURIER_VSTRIDE;
				t0 = FOURIER_VADD(a, b);
				t1 = FOURIER_VSUB(a, b);
				FOURIER_VINTERLEAVE(t0, t1, &a, &b);
				FOURIER_VSTOREU(Dst, a); Dst += FOURIER_VSTRIDE;
				FOURIER_VSTOREU(Dst, b); Dst += FOURIER_VSTRIDE;
			}
		}
#else
		i = 0;
#endif
		float a, b;
		for(;i<N/2-1;i++) {
			a = *TmpLo++;
			b = *--TmpHi;
			*Dst++ = a + b;
			*Dst++ = a - b;
		}
		*Dst++ = *--TmpHi;
	}
}

// Implementation notes for IMDCT:
//  IMDCT is implemented via DCT-IV, which can be thought of
//  as splitting the MDCT inputs into four regions:
//   {A,B,C,D}
//  and then taking the DCT-IV of:
//   {C_r + D, B_r - A}
//  On IMDCT, we get back these latter values following an
//  inverse DCT-IV (which is itself a DCT-IV due to its
//  involutive nature).
//  From a prior call, we keep C_r+D buffered, which becomes
//  A_r+B after accounting for movement to the next block.
//  We can then state:
//   Reverse(A_r + B) -		(B_r - A) = (A + B_r) - (B_r - A) = 2A
//		  (A_r + B) + Reverse(B_r - A) = (A_r + B) + (B - A_r) = 2B
//		   ^ Buffered		 ^ New input data
//  Allowing us to reconstruct the inputs A,B.

FOURIER_FORCE_INLINE void Fourier_IMDCT(float *BufOut, const float *BufIn, float *BufLap, float *BufTmp, const int N, int Overlap) {
	int i;
	FOURIER_ASSUME_ALIGNED(BufOut, FOURIER_ALIGNMENT);
	FOURIER_ASSUME_ALIGNED(BufIn,  FOURIER_ALIGNMENT);
	FOURIER_ASSUME_ALIGNED(BufLap, FOURIER_ALIGNMENT);
	FOURIER_ASSUME_ALIGNED(BufTmp, FOURIER_ALIGNMENT);
	FOURIER_ASSUME(N >= 16);
	FOURIER_ASSUME(Overlap >= 0 && Overlap <= N);

	const float *Lap   = BufLap + N/2;
	const float *Tmp   = BufTmp + N/2;
		  float *OutLo = BufOut;
		  float *OutHi = BufOut + N;

	// Undo transform
	memcpy(BufTmp, BufIn, N * sizeof(float));
	// for(i=0;i<N;i++) BufTmp[i] = BufIn[i];
	Fourier_DCT4(BufTmp, BufOut, N);
	
	// Undo lapping
#if FOURIER_VSTRIDE > 1
	Fourier_Vec_t a, b;
	for(i=0;i<(N-Overlap)/2;i+=FOURIER_VSTRIDE) {
		Lap -= FOURIER_VSTRIDE; a = FOURIER_VLOAD(Lap);
		b = FOURIER_VLOAD(Tmp); Tmp += FOURIER_VSTRIDE;
		a = FOURIER_VREVERSE(a);
		b = FOURIER_VREVERSE(b);
		FOURIER_VSTORE(OutLo, a); OutLo += FOURIER_VSTRIDE;
		OutHi -= FOURIER_VSTRIDE; FOURIER_VSTORE(OutHi, b);
	}
	Fourier_Vec_t t0, t1 = FOURIER_VMUL(FOURIER_VSET1(1.0f/Overlap), FOURIER_VADD(FOURIER_VSET_LINEAR_RAMP(), FOURIER_VSET1(0.5f)));
	Fourier_Vec_t c  = Fourier_Cos(t1);
	Fourier_Vec_t s  = Fourier_Sin(t1);
	Fourier_Vec_t wc = Fourier_Cos(FOURIER_VSET1((float)FOURIER_VSTRIDE / Overlap));
	Fourier_Vec_t ws = Fourier_Sin(FOURIER_VSET1((float)FOURIER_VSTRIDE / Overlap));
	for(;i<N/2;i+=FOURIER_VSTRIDE) {
		Lap -= FOURIER_VSTRIDE; a = FOURIER_VREVERSE(FOURIER_VLOAD(Lap));
		b  = FOURIER_VLOAD(Tmp); Tmp += FOURIER_VSTRIDE;
		t0 = FOURIER_VFMS(c, a, FOURIER_VMUL(s, b));
		t1 = FOURIER_VFMA(s, a, FOURIER_VMUL(c, b));
		t1 = FOURIER_VREVERSE(t1);
		FOURIER_VSTORE(OutLo, t0); OutLo += FOURIER_VSTRIDE;
		OutHi -= FOURIER_VSTRIDE; FOURIER_VSTORE(OutHi, t1);
		t0 = c;
		t1 = s;
		c = FOURIER_VNFMA(t1, ws, FOURIER_VMUL(t0, wc));
		s = FOURIER_VFMA (t1, wc, FOURIER_VMUL(t0, ws));
	}
#else

	for(i=0; i<(N-Overlap)/2; i++) {
		*OutLo++ = *--Lap;
		*--OutHi = *Tmp++;
	}

	if (Overlap > 0) {
		float c  = Fourier_Cos(0.5f / Overlap);
		float s  = Fourier_Sin(0.5f / Overlap);
		float wc = Fourier_Cos(1.0f / Overlap);
		float ws = Fourier_Sin(1.0f / Overlap);
		for(;i<N/2;i++) {
			float a = *--Lap;
			float b = *Tmp++;
			*OutLo++ = c*a - s*b;
			*--OutHi = s*a + c*b;
			a = c;
			b = s;
			c = wc*a - ws*b;
			s = ws*a + wc*b;
		}
	}
#endif
	// Copy state to old block
	memcpy(BufLap, BufTmp, (N/2) * sizeof(float));
}







/**************************************/
#define BUFFER_ALIGNMENT 64u //! Always align memory to 64-byte boundaries (preparation for AVX-512)
/**************************************/

//! Just for consistency
#define MIN_CHANS	 1
#define MAX_CHANS   255
#define MIN_BANDS   256
#define MAX_BANDS 32768

/**************************************/

//! Initialize decoder state
int ULC_DecoderState_Init(struct ULC_DecoderState_t *State) {
	//! Clear anything that is needed for EncoderState_Destroy()
	State->BufferData  = NULL;

	//! Verify parameters
	int nChan	 = State->nChan;
	int BlockSize = State->BlockSize;
	if(nChan	 < MIN_CHANS || nChan	 > MAX_CHANS) return -1;
	if(BlockSize < MIN_BANDS || BlockSize > MAX_BANDS) return -1;
	if((BlockSize & (-BlockSize)) != BlockSize)		return -1;

	//! Get buffer offsets and allocation size
	int AllocSize = 0;
#define CREATE_BUFFER(Name, Sz) int Name##_Offs = AllocSize; AllocSize += Sz
	CREATE_BUFFER(TransformBuffer, sizeof(float) * (	   BlockSize   ));
	CREATE_BUFFER(TransformTemp,   sizeof(float) * (nChan* BlockSize   ) * 2);
	CREATE_BUFFER(TransformInvLap, sizeof(float) * (nChan*(BlockSize/2)));
#undef CREATE_BUFFER

	//! Allocate buffer space
	char *Buf = State->BufferData = malloc(BUFFER_ALIGNMENT-1 + AllocSize);
	if(!Buf) return -1;

	//! Initialize state
	int i;
	Buf += (-(uintptr_t)Buf) & (BUFFER_ALIGNMENT-1);
	State->LastSubBlockSize = 0;
	State->TransformBuffer = (float*)(Buf + TransformBuffer_Offs);
	State->TransformTemp   = (float*)(Buf + TransformTemp_Offs);
	State->TransformInvLap = (float*)(Buf + TransformInvLap_Offs);
	for(i=0;i<nChan*(BlockSize/2);i++) State->TransformInvLap[i] = 0.0f;

	//! Success
	return 1;
}

/**************************************/

//! Destroy decoder state
void ULC_DecoderState_Destroy(struct ULC_DecoderState_t *State) {
	//! Free buffer space
	free(State->BufferData);
}

/**************************************/

//! Decode block
#define ESCAPE_SEQUENCE_STOP		   (-1)
#define ESCAPE_SEQUENCE_STOP_NOISEFILL (-2)
static inline uint32_t Block_Decode_UpdateRandomSeed(void) {
	static uint32_t Seed = 1234567;
	Seed ^= Seed << 13; //! Xorshift
	Seed ^= Seed >> 17;
	Seed ^= Seed <<  5;
	return Seed;
}
static inline uint8_t Block_Decode_ReadNybble(const uint8_t **Src, int *Size) {
	//! Fetch and shift nybble
	uint8_t x = *(*Src);
	*Size += 4;
	if((*Size)%8u == 0) x >>= 4, (*Src)++;
	return x&0xF;
}
static inline int Block_Decode_ReadQuantizer(const uint8_t **Src, int *Size) {
	int		   qi  = Block_Decode_ReadNybble(Src, Size); //! Fh,0h..Dh:	  Quantizer change
	if(qi == 0xF) return ESCAPE_SEQUENCE_STOP_NOISEFILL;	//! Fh,Fh,Zh,Yh,Xh: Noise fill (to end; exp-decay)
	if(qi == 0xE) qi += Block_Decode_ReadNybble(Src, Size); //! Fh,Eh,0h..Ch:   Quantizer change (extended precision)
	if(qi == 0xE + 0xF) return ESCAPE_SEQUENCE_STOP;		//! Fh,Eh,Fh:	   Zeros fill (to end)
	return qi;
}
static inline float Block_Decode_ExpandQuantizer(int qi) {
	return 0x1.0p-31f * ((1u<<(31-5)) >> qi); //! 1 / (2^5 * 2^qi)
}
static inline int Block_Decode_DecodeSubBlockCoefs(float *CoefDst, int N, const uint8_t **Src, int *Size) {
	int32_t n, v;

	//! Check first quantizer for Stop code
	v = Block_Decode_ReadQuantizer(Src, Size);
	if(v == ESCAPE_SEQUENCE_STOP) {
		//! [Fh,]Eh,Fh: Stop
		do *CoefDst++ = 0.0f; while(--N);
		return 1;
	}

	//! Unpack the [sub]block's coefficients
	float Quant = Block_Decode_ExpandQuantizer(v);
	for(;;) {
		//! -7h..-2h, +2..+7h: Normal
		v = Block_Decode_ReadNybble(Src, Size);
		if(v != 0x0 && v != 0x1 && v != 0x8 && v != 0xF) { //! <- Exclude all control codes
			//! Store linearized, dequantized coefficient
			v = (v^0x8) - 0x8; //! Sign extension
			v = (v < 0) ? (-v*v) : (+v*v);
			*CoefDst++ = v * Quant;
			if(--N == 0) break;
			continue;
		}

		//! 0h,0h..Fh: Zeros fill (1 .. 16 coefficients)
		if(v == 0x0) {
			n = Block_Decode_ReadNybble(Src, Size) + 1;
			if(n > N) return 0;
			N -= n;
			do *CoefDst++ = 0.0f; while(--n);
			if(N == 0) break;
			continue;
		}

		//! 1h,Yh,Xh: 33 .. 288 zeros fill
		if(v == 0x1) {
			n  = Block_Decode_ReadNybble(Src, Size);
			n  = Block_Decode_ReadNybble(Src, Size) | (n<<4);
			n += 33;
			if(n > N) return 0;
			N -= n;
			do *CoefDst++ = 0.0f; while(--n);
			if(N == 0) break;
			continue;
		}

		//! 8h,Zh,Yh,Xh: 16 .. 527 noise fill
		if(v == 0x8) {
			n  = Block_Decode_ReadNybble(Src, Size);
			n  = Block_Decode_ReadNybble(Src, Size) | (n<<4);
			v  = Block_Decode_ReadNybble(Src, Size);
			n  = (v&1) | (n<<1);
			v  = (v>>1) + 1;
			n += 16;
			if(n > N) return 0;
			N -= n; {
				float p = (v*v) * Quant * (1.0f/4);
				do {
					if(Block_Decode_UpdateRandomSeed() & 0x80000000) p = -p;
					*CoefDst++ = p;
				} while(--n);
			}
			if(N == 0) break;
			continue;
		}

		//! Fh,0h..Dh:	Quantizer change
		//! Fh,Eh,0h..Ch: Quantizer change (extended precision)
		v = Block_Decode_ReadQuantizer(Src, Size);
		if(v >= 0) {
			Quant = Block_Decode_ExpandQuantizer(v);
			continue;
		}

		//! Fh,Fh,Zh,Yh,Xh: Noise fill (to end; exp-decay)
		if(v == ESCAPE_SEQUENCE_STOP_NOISEFILL) {
			v = Block_Decode_ReadNybble(Src, Size) + 1;
			n = Block_Decode_ReadNybble(Src, Size);
			n = Block_Decode_ReadNybble(Src, Size) | (n<<4);
			float p = (v*v) * Quant * (1.0f/16);
			float r = 1.0f + (n*n)*-0x1.0p-19f;
			do {
				if(Block_Decode_UpdateRandomSeed() & 0x80000000) p = -p;
				*CoefDst++ = p, p *= r;
			} while(--N);
			break;
		}

		//! Fh,Eh,Dh: Unused
		//! Fh,Eh,Eh: Unused
		//! Fh,Eh,Fh: Zeros fill (to end)
		if(v == ESCAPE_SEQUENCE_STOP) {
			do *CoefDst++ = 0.0f; while(--N);
			break;
		}
	}
	return 1;
}
int ULC_DecodeBlock(struct ULC_DecoderState_t *State, float *DstData, const void *_SrcBuffer) {
	//! Spill state to local variables to make things easier to read
	int	n;
	int	nChan		   = State->nChan;
	int	BlockSize	   = State->BlockSize;
	float *TransformBuffer = State->TransformBuffer;
	float *TransformTemp   = State->TransformTemp;
	float *TransformInvLap = State->TransformInvLap;
	const uint8_t *SrcBuffer = _SrcBuffer;

	//! Begin decoding
	int Chan, Size = 0;
	int LastSubBlockSize = 0; //! <- Shuts gcc up
	int WindowCtrl; {
		//! Read window control information
		WindowCtrl = Block_Decode_ReadNybble(&SrcBuffer, &Size);
		if(WindowCtrl & 0x8) WindowCtrl |= Block_Decode_ReadNybble(&SrcBuffer, &Size) << 4;
		else				 WindowCtrl |= 1 << 4;
	}
	for(Chan=0;Chan<nChan;Chan++) {
		//! Reset overlap scaling for this channel
		LastSubBlockSize = State->LastSubBlockSize;

		//! Process subblocks
		float *Dst = DstData + Chan*BlockSize;
		float *Src = TransformBuffer;
		float *Lap = TransformInvLap;
		ULC_SubBlockDecimationPattern_t DecimationPattern = ULCi_SubBlockDecimationPattern(WindowCtrl);
		do {
			int SubBlockSize = BlockSize >> (DecimationPattern&0x7);
			if(!Block_Decode_DecodeSubBlockCoefs(Src, SubBlockSize, &SrcBuffer, &Size)) {
				//! Corrupt block
				return 0;
			}

			//! Get+update overlap size and limit to that of the last subblock
			int OverlapSize = SubBlockSize;
			if(DecimationPattern&0x8)
				OverlapSize >>= (WindowCtrl & 0x7);
			if(OverlapSize > LastSubBlockSize)
				OverlapSize = LastSubBlockSize;
			LastSubBlockSize = SubBlockSize;

			//! A single long block can be read straight into the output buffer
			if(SubBlockSize == BlockSize) {
				Fourier_IMDCT(Dst, Src, Lap, TransformTemp, SubBlockSize, OverlapSize);
				break;
			}

			//! For small blocks, we store the decoded data to a scratch buffer
			float *DecBuf = TransformTemp + SubBlockSize;
			Fourier_IMDCT(DecBuf, Src, Lap, TransformTemp, SubBlockSize, OverlapSize);

			//! Output samples from the lapping buffer, and cycle
			//! the new samples through it for the next call
			int nAvailable = (BlockSize - SubBlockSize) / 2;
				  float *LapDst = Lap + BlockSize/2;
			const float *LapSrc = LapDst;
			if(SubBlockSize <= nAvailable) {
				//! We have enough data in the lapping buffer
				//! to output a full subblock directly from it,
				//! so we do that and then shift any remaining
				//! data before re-filling the buffer.
				for(n=0;n<SubBlockSize;n++) *Dst++	= *--LapSrc;
				for(   ;n<nAvailable  ;n++) *--LapDst = *--LapSrc;
				for(n=0;n<SubBlockSize;n++) *--LapDst = *DecBuf++;
			} else {
				//! We only have enough data for a partial output
				//! from the lapping buffer, so output what we can
				//! and output the rest from the decoded buffer
				//! before re-filling.
				for(n=0;n<nAvailable;  n++) *Dst++	= *--LapSrc;
				for(   ;n<SubBlockSize;n++) *Dst++	= *DecBuf++;
				for(n=0;n<nAvailable;  n++) *--LapDst = *DecBuf++;
			}
		} while(DecimationPattern >>= 4);

		//! Move to next channel
		TransformInvLap += BlockSize/2;
	}

	//! Undo M/S transform
	//! NOTE: Not orthogonal; must be fully normalized on the encoder side.
	if(nChan != 1) {
		for(int i = 0, j = 0; i < BlockSize; i++, j += 2) {
			float a = DstData[i];
			float b = DstData[i + BlockSize];
			TransformTemp[j] = a + b;
			TransformTemp[j+1] = a - b;
		}	
		memcpy(DstData, TransformTemp, sizeof(float) * BlockSize*nChan);
	}

	//! Store the last [sub]block size, and return the number of bits read
	State->LastSubBlockSize = LastSubBlockSize;
	return Size;
}

/**************************************/
//! EOF
/**************************************/
