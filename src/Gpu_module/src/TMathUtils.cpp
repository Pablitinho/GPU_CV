/*
 * TMathUtils.cpp
 *
 *  Created on: 27/02/2015
 *      Author: pablo
 */

#include "TMathUtils.h"
#include <math.h>
//--------------------------------------------------------------------------

typedef union half_bits half_bits;
union half_bits
{
	float f;
	unsigned int i;
	unsigned short s[2];
	unsigned char c[4];
};

//Macros for portability...
#if __LITTLE_ENDIAN__
	#define BP_LEFT		1
	#define BP_RIGHT	0
#else
	#define BP_LEFT		0
	#define BP_RIGHT	1
#endif
//--------------------------------------------------------------------------
TMathUtils::TMathUtils()
{


}
//--------------------------------------------------------------------------
float TMathUtils::HalftoFloat(unsigned short in_half)
{
	short mantissa = (in_half & 0x7C00) >> 10;

	if (mantissa == 0)
		return 0;

	if (mantissa == 0x001F)
	{
		if (in_half & 0x8000)
			return -INFINITY;
		return INFINITY;
	}

	mantissa += 127 - 16;

	half_bits hb;
	hb.s[BP_LEFT] =		(in_half & 0x8000)				//Sign bit		(1 bits)
				|	((mantissa << 7) & 0x7F80)			//Mantissa		(8 bits)
				|	((in_half >> 3) & 0x007F);			//Significand	(7 bits)
	hb.s[BP_RIGHT] =		(in_half << 13) & 0xFFFF;	//Significand	(16 bits)

	return hb.f;
}
//--------------------------------------------------------------------------
TMathUtils::~TMathUtils()
{

}

