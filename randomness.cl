
/* Generates pseudo-random floating points numbers in [-0.5,0.5). */
float random_uniform(unsigned int state[static const 5]) {
    //The state contains the current state of a pseudo-random number generator
	//this operation below uses a variant called XORWOW
    unsigned int s,
	             t = state[3];
	t ^= t >> 2;
	t ^= t << 1;
	state[3] = state[2]; state[2] = state[1]; state[1] = s = state[0];
	t ^= s;
	t ^= s << 4;
	state[0] = t;
	state[4] += 362437;

	//t + state[4] should now contain 32 bits of pseudorandomness.
	//It's not unusual to compute something like (float)(t + state[4])/)(float)INT_MAX), but this
	//requires division. Another way to create a random number is to use bit operations to turn something
	//directly into a float

	//A 32-bit floating-point number has a sign bit in the highest position. Sice we are to generate a positive number
	//this is to be zero. Then there are eight bits of exponent and 23 bits for the fractional part. We will fill
	//the fractional part with bits of randomness from t + state[4] by shifting it to create zeros in the nine highest
	//bits. We then add the appropriate exponent and sign bits using bitwise or with the hexadecimal representation of 1.0f.
	//This produces a number that has the exponent of 1.0f, but a bunch of decimals instead of, up to something like 1.999.
	//Finally 1.5 is subtracted so as to obtain a number in [-0.5,0.5).

	return as_float(0x3F800000 | t + state[4] >> 9) - 1.5f;
}

/* Modification of the Marsaglia polar method for generating normal-distributed random variables
   replacing the square with a hexagon */
float2 random_normal1(unsigned int state[static const 5]) {
    const float c = 3.0f/sqrt(3.0f),
	            d = 0.5f/sqrt(3.0f),
				e = sqrt(3.0f),
				f = 2.0f/sqrt(3.0f);
    float x, y, s;
	do {
    	y = random_uniform(state)*2.0f;
		x = random_uniform(state)*c - d;
		if(fabs(y) > (x + f)*e) {
		    x = x + c;
			y = sign(y)*(1-fabs(y));
		}
		s = x*x + y*y;
	} while(s > 1 || s == 0.0f);
	s = sqrt(-2.0*log(s)/s);
	return (float2)(x*s, y*s);
}

/* Marsaglia polar method */
float2 random_normal2(unsigned int state[static const 5]) {
   float x, y, s;
   do {
       x = random_uniform(state)*2.0f;
	   y = random_uniform(state)*2.0f;
	   s = x*x + y*y;
   } while(s > 1 || s == 0.0f);
   s = sqrt(-2.0*log(s)/s);
   return (float2)(x*s, y*s);
}

/* Box-Muller transform */
float2 random_normal3(unsigned int state[static const 5]) {
    float x, y, s;
	do {
	    x = random_uniform(state) + 0.5f;
		y = random_uniform(state) + 0.5f;
	} while(x == 0.0f);
	s = sqrt(-2.0f*log(x));
	return (float2)(s*cos(2.0f*M_PI_F * y), s*sin(2.0f*M_PI_F * y));
}

/* Verify properties of the distributions */
__kernel void test_moment(__global float* output, int p, int mode, int N) {
    const int id = get_global_id(0);
    unsigned int state[5];
	state[0] = id;
	int i;
	float2 sums = 0.0f;
	for(i = 0; i != N; ++i) {
        float2 output;
        if(mode == 1) output = random_normal1(state);
		else if(mode == 2) output = random_normal2(state);
		else if(mode == 3) output = random_normal3(state);
		sums += (pown(output, p)/(float)N);
	}
	output[id] = dot(sums, (float2)(0.5f, 0.5f));
}

/* Verify further properties of the distributions */
__kernel void test_correlation(__global float* output, int mode, int N) {
    const int id = get_global_id(0);
	unsigned int state[5];
	state[0] = id;
	int i;
	float sum = 0.0f;
	for(i = 0; i != N; ++i) {
	    float2 output;
		if(mode == 1) output = random_normal1(state);
		else if(mode == 2) output = random_normal2(state);
		else if(mode == 3) output = random_normal3(state);
		sum += output.x * output.y;
	}
	output[id] = sum/(float)N;
}

/* Speed benchmark the generators for normal-distributed random variables */
__kernel void speed_test(int mode, int N) {
    unsigned int state[5];
	int i;
	if(mode == 1)
    	for(i = 0; i != N; ++i)
	        random_normal1(state);
	else if(mode == 2)
	    for(i = 0; i != N; ++i)
		    random_normal2(state);
	else if(mode == 3)
	    for(i = 0; i != N; ++i)
		    random_normal3(state);
}