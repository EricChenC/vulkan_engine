#version 450
#extension GL_ARB_separate_shader_objects : enable

// data from vertex shader
layout(location = 1) in vec2 outTexcoord;
layout(location = 2) in vec3 outNormal;
layout(location = 3) in vec3 outLightPos;
layout(location = 4) in vec3 outFragPos;	// out view
layout(location = 5) in vec3 outPosition;

// special
layout(binding = 6) uniform UniformSpecialParameters{
	// basic parameters
	vec4  coatColor;
	vec4  coatSpecularColor;
	vec4  baseColor;
	
	float bumpSize;
	float bumpScale;
	float coatOpacity;
	float coatShininess;
	float coatReflectivity;
	float baseShininess;
	float baseReflectivity;
}usp;

//uniform sampler2D u_texReflection[7];

layout(location = 0) out vec4 outColor;

// Atom method
// color luminance
float luminance(vec3 col)
{
	return dot(col, vec3(0.212671, 0.715160, 0.072169));
}

// Atom method
// Convert color-normal to regular normal for normal mapping
vec3 color2Normal(vec3 col)
{
	return normalize(col - vec3(0.5));
}

// Atom method
// gamma correction
vec4 gammaCorrection(float fGamma, vec4 col)
{
	vec4 vFactor = vec4(1.0 / fGamma);
	return pow(col, vFactor);
}

// Atom method
// a simple heck to smooth the transition of lit boundary
float fallOff(float a, float b, float x)
{
	if (x < a)  return 0.0;
	if (x >= b) return 1.0;
	x = (x - a) / (b - a);
	
	return x * x * (3.0 - 2.0 * x);
}

// Atom method
// Precise Fresnel computation for dielectric surface
float fresnelDielectric (float NdotV, float n1, float n2)
{
	float cosi = clamp(NdotV, 0.0, 1.0);
	//float cosi = NdotV;
	//Snell's law
	float sint = n1 / n2 * sqrt(1.0 - cosi*cosi);
	if (sint > 1.0)
	{
		return 1.0;
	}

	float cost = sqrt(1.0 - sint*sint);
	float Rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
	float Rp = (n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi);

	return (Rs * Rs + Rp * Rp) / 2.0;
}


// Atom method
// diffuse computation using Oren-Nayar model
float diffuseOrenNayar(float fRough, vec3 vN, vec3 vL, vec3 vV, float NdotL, float NdotV)
{
	float sinI = sqrt(1.0 - NdotL * NdotL);
	float sinO = sqrt(1.0 - NdotV * NdotV);

	// Compute cosine term of Oren--Nayar model
	float maxCos = 0.0;
	if ((sinI > 0.0) && (sinO > 0.0))
	{
		vec3 normalWithLight = vN * NdotL;
		vec3 vecWithNormalLight = normalize(vL - normalWithLight);
		vec3 normalWithView = vN * NdotV;
		vec3 vecWithNormalView = normalize(vV - normalWithView);
		float dCos = dot(vecWithNormalLight, vecWithNormalView);
		maxCos = max(0.0, dCos);
	}
	// Compute sine and tangent terms of Oren--Nayar model
	float sinA;
	float tanB;
	if (NdotL > NdotV)
	{
		sinA = sinO;
		tanB = sinI / NdotL;
	}
	else
	{
		sinA = sinI;
		tanB = sinO / NdotV;
	}

	float r2 = fRough * fRough;
	float fA = 1.0 - (r2 / (2.0 * (r2 + 0.33)));	//A = 1.0f - (r^2 / (2.0f * (r^2 + 0.33f)));
	float fB = 0.45 * r2 / (r2 + 0.09);				//B = 0.45f * r^2 / (r^2 + 0.09f);

	return (fA + fB * maxCos * sinA * tanB);
}


// Atom method
// An improved Ward model for specular computation
float specularWardAdv(float NdotL, float NdotV, float NdotH, float HdotXY, float shininess)
{
	float factor = NdotL * NdotV;
	if (factor < 0.000001)
		return 0.0;

	float sh2 = shininess * shininess;
	float NH2 = NdotH * NdotH;
	float foo = HdotXY * sh2;
	if (NH2 < 0.00001 * foo)
	{
		return 0.0;
	}

	float ret = 0.0;
	float expo = -foo / NH2;
	vec3 lobs = exp(vec3(expo, expo * 0.25, expo * 0.0625));
	ret += 0.5 * lobs.x;
	if (NH2 >= 0.000025 * foo)
	{
		ret += 0.25 * lobs.y;
		if (NH2 >= 0.00000625 * foo)
			ret += 0.09375 * lobs.z;
	}

	return ret * sh2 / (sqrt(factor) * 12.56637);
}

// Atom method
// unit vector to spherical coordinate and scaled to [0, 1] for
// reflection and background texture lookup
vec2 vector2ThetaPhi(vec3 v)
{
	float theta = atan(v.x, v.z);
	float phi = 3.1415926 * 0.5 - acos(v.y);
	return vec2((theta + 3.1415926) * (0.159155), 1.0 - 0.5 * (1.0 + sin(phi)));
}


vec3 permute(vec3 x0, vec3 p)
{
	vec3 x1 = mod(x0 * p.y, p.x);
	return floor(mod((x1 + p.z) *x0, p.x));
}
vec4 permute(vec4 x0, vec3 p)
{
	vec4 x1 = mod(x0 * p.y, p.x);
	return floor(mod((x1 + p.z) *x0, p.x));
}

#define taylorInvSqrt(r) (0.83666002653408 + 0.7*0.85373472095314 - 0.85373472095314 * r)

float perlinNoise3D(vec3 v)
{
	const vec2  C = vec2(1. / 6., 1. / 3.);
	const vec4  D = vec4(0., 0.5, 1.0, 2.0);
	const vec4  pParam = vec4(17.0*17.0, 34.0, 1.0, 7.0);

	// First corner
	vec3 i = floor(v + dot(v, C.yyy));
	vec3 x0 = v - i + dot(i, C.xxx);

	// Other corners
	vec3 g = vec3(greaterThan(x0.xyz, x0.yzx));
	vec3 l = vec3(lessThanEqual(x0.xyz, x0.yzx));

	vec3 i1 = g.xyz  * l.zxy;
	vec3 i2 = max(g.xyz, l.zxy);

	//  x0 = x0 - 0. + 0. * C 
	vec3 x1 = x0 - i1 + 1. * C.xxx;
	vec3 x2 = x0 - i2 + 2. * C.xxx;
	vec3 x3 = x0 - 1. + 3. * C.xxx;

	// Permutations
	i = mod(i, pParam.x);
	vec4 p = permute(permute(permute(
		i.z + vec4(0., i1.z, i2.z, 1.), pParam.xyz)
		+ i.y + vec4(0., i1.y, i2.y, 1.), pParam.xyz)
		+ i.x + vec4(0., i1.x, i2.x, 1.), pParam.xyz);

	// Gradients
	// ( N*N points uniformly over a square, mapped onto a octohedron.)
	float n_ = 1.0 / pParam.w;
	vec3  ns = n_ * D.wyz - D.xzx;

	vec4 j = p - pParam.w*pParam.w*floor(p * ns.z *ns.z);  //  mod(p,N*N)

	vec4 x_ = floor(j * ns.z);
	vec4 y_ = floor(j - pParam.w * x_);    // mod(j,N)

	vec4 x = x_ *ns.x + ns.yyyy;
	vec4 y = y_ *ns.x + ns.yyyy;
	vec4 h = 1. - abs(x) - abs(y);

	vec4 b0 = vec4(x.xy, y.xy);
	vec4 b1 = vec4(x.zw, y.zw);

	vec4 s0 = vec4(lessThan(b0, D.xxxx)) * 2. - 1.;
	vec4 s1 = vec4(lessThan(b1, D.xxxx)) * 2. - 1.;
	vec4 sh = vec4(lessThan(h, D.xxxx));

	vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
	vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;

	vec3 p0 = vec3(a0.xy, h.x);
	vec3 p1 = vec3(a0.zw, h.y);
	vec3 p2 = vec3(a1.xy, h.z);
	vec3 p3 = vec3(a1.zw, h.w);

	p0 *= taylorInvSqrt(dot(p0, p0));
	p1 *= taylorInvSqrt(dot(p1, p1));
	p2 *= taylorInvSqrt(dot(p2, p2));
	p3 *= taylorInvSqrt(dot(p3, p3));

	// Mix
	vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.);
	m = m * m;
	return 64.0 * dot(m*m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

float orangePeelBump(vec3 p)
{
	p = p * 20 / usp.bumpSize;
	return  perlinNoise3D (p) * 0.5 + 0.5;
}

// main routine
void main()
{
	const float esp = 0.0001;
	// basic vectors
	vec3 vL = normalize(vec3(1.0));
	vec3 vV = normalize(outFragPos);
	vec3 vN = normalize(outNormal);
	vec4 L = vec4(1.0);	// incident light color

	// falloff heck
	// This code should be place before the normal being modified by bump mapping
	// SNdorL: surface-normal-dot-lightdirection
	float SNdotL = max(dot(vN, vL), 0.0);
	float fFallOff = fallOff(0.0f, 0.25f, SNdotL);
	
	// orange peel bump
	if (usp.bumpScale >= 0.01)
	{
		float g0 = orangePeelBump(outPosition);
		float gx = orangePeelBump(outPosition + vec3(esp, 0.f, 0.f));
		float gy = orangePeelBump(outPosition + vec3(0.f, esp, 0.f));
		float gz = orangePeelBump(outPosition + vec3(0.f, 0.f, esp));
		float scale = 5.0 * usp.bumpScale;
		vN = normalize(vN - vec3(gx - g0, gy - g0, gz - g0) * scale);
	}

	// balance energy
	// balance reflection and transmission for coat
	float NdotV = max(dot(-vV, vN), 0.0);
	// select one of the following fresnel method
	//float reflFactor = fresnelDielectric(NdotV, u_IoR, 1.0);
	float coatReflFactor = fresnelDielectric(NdotV, 1.f, 3.0);
	coatReflFactor = usp.coatReflectivity + (1.0 - usp.coatReflectivity) * coatReflFactor;
	//float reflFactor = fresnelConductor (NdotV, eta, extinction)
	float coatTranFactor = 1.f - coatReflFactor;
	// balance reflection and transmission for base	
	float baseReflFactor = fresnelDielectric(NdotV, 1.f, 1.5);
	baseReflFactor = usp.baseReflectivity + (1.0 - usp.baseReflectivity) * baseReflFactor;
	//float reflFactor = fresnelConductor (NdotV, eta, extinction)
	float baseTranFactor = (1.f - baseReflFactor) * coatTranFactor;
	baseReflFactor = baseReflFactor * coatTranFactor;
	
	// the reflection direction
	vec3 vR = normalize(reflect(vV, vN));
	// coat attenuations according to different angles
	float coatTransparency = 1 - usp.coatOpacity;
	float coatTranFactorV = pow(coatTransparency, 1.0 / max(NdotV, 0.04));
	float coatTranFactorL = pow(coatTransparency, 1.0 / max(dot(vN, vL), 0.04));
	float coatTranFactorR = pow(coatTransparency, 1.0 / max(dot(vN, vR), 0.04));

	//--- calculate the reflection contribution using an environment map
	// requires "uniform sampler2D u_texReflection[7];"
	// this is for ward-like specular
	// specular reflection
	float shininess = usp.coatShininess;
	vec4 reflection = vec4(0.25);
	float fT;
	if (shininess > 0)
	{
		int idx;
		if (shininess <= 1.0)			// l = 6, 5
		{
			idx = 5;
			fT = 1.0 - shininess;
		}
		else if (shininess <= 4.0)		// l = 5, 4
		{
			idx = 4;
			fT = (4.0 - shininess) / 3.0; // 4 - 1 = 3
		}
		else if (shininess <= 16.0)		// l = 4, 3
		{
			idx = 3;
			fT = (16.0 - shininess) / 12.0; // 16 - 4 = 12
		}
		else if (shininess <= 48.0)		// l = 3, 2
		{
			idx = 2;
			fT = (48.0 - shininess) / 32.0; // 48 - 16 = 32
		}
		else if (shininess <= 100.0)	// l = 2, 1
		{
			idx = 1;
			fT = max((100 - shininess) / 100.0, 0.0);
		}
		
		//fT = pow(fT, 0.8);
		//vec4 spec0 = texture2D(u_texReflection[idx-1], vector2ThetaPhi(vR)) * (1.0 - fT);
		//vec4 spec1 = texture2D(u_texReflection[idx], vector2ThetaPhi(vR)) * fT;
		
		//reflection = (spec1 + spec0);
	}
	
	vec3 vH = normalize(-vV + vL);

	float NdotL = max(dot(vL, vN), 0.0);
	float NdotH = max(dot(vN, vH), 0.0);
	float HdotXY = sqrt(1.0 - NdotH * NdotH);

	// component reflectivities
	float fCoatSpecCoef = specularWardAdv(NdotL, NdotV, NdotH, HdotXY, usp.coatShininess);
	float fBaseSpecCoef = specularWardAdv(NdotL, NdotV, NdotH, HdotXY, usp.baseShininess);
	float fBaseDiffCoef = diffuseOrenNayar(0.5, vN, vL, vV, NdotL, NdotV) * baseTranFactor;

	// coat specular reflection
	vec4 vCoatSpec = vec4(vec3(usp.coatSpecularColor), 1.0) * (L * fCoatSpecCoef * fFallOff + reflection) * coatReflFactor;
	vec4 vCoatDiff = vec4(vec3(usp.coatColor), 1.0) * (L * fCoatSpecCoef * fFallOff + reflection) * 
						((1 - coatTranFactorV) * coatTranFactor);
	// color reach base
	vec4 vBaseLit = L * coatTranFactorL * fFallOff;
	vec4 vBaseEnvLit = reflection * coatTranFactorR;
	
	float fBaseSpec = baseReflFactor * fBaseSpecCoef;
	float fBaseDiff = baseTranFactor * fBaseDiffCoef;
	vec4 vBase = vec4(vec3(fBaseSpec) + vec3(usp.baseColor) * fBaseDiff, 1.0) * vBaseLit + 
					vec4(vec3(baseReflFactor) + vec3(usp.baseColor) * fBaseDiffCoef, 1.0) * vBaseEnvLit;
	vBase = vBase * coatTranFactorV;
	
	outColor = gammaCorrection(2.2, vCoatSpec + vCoatDiff + vBase);
	outColor.a = 1.0;
}