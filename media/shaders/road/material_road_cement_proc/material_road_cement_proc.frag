#version 450
#extension GL_ARB_separate_shader_objects : enable

// data from vertex shader
layout(location = 1) in vec2 outTexcoord;
layout(location = 2) in vec3 outNormal;
layout(location = 3) in vec3 outLightPos;
layout(location = 4) in vec3 outFragPos;	// out view
layout(location = 5) in vec3 outPosition;

// normal 
layout(binding = 1) uniform UniformNormalParameters{
	vec4 ambientColor;						// ambient color
	vec4 diffuseColor;						// diffuse color
	vec4 specularColor;						// specular color
	vec4 transparency;						// transparency
	
	float diffuseRough;						// diffuse roughness
	float shininess;						// specular shininess
	float reflectivity;						// specular reflectivity
	float indexOfRefraction;				// index of refraction
	float extinction;						// extinction of metal
	float opacity;
	
	uint  options;
	uint  version;							// shader version
}unp;

layout(binding = 6) uniform UniformSpecialParameters{
	float sizeScale;				// size scale
}usp;

// textures
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
// for Schlick Fresnel computation
float pow5(float x)
{
	float foo = x * x;
	return foo * foo * x;
}

// Atom method
// Schlick Fresnel approximation
// Require pow5() method
float fresnelSchlick(float NdotV, float fReflectivity)
{
	if (fReflectivity >= 1.0) return 1.0;
	return fReflectivity + (1 - fReflectivity) * pow5(1.0 - NdotV);
}


// Atom method
// a better phong specular model
// the three vectors, V, L and N are viewing, lighting directions and surface normal
// sh: shininess in [0, 99]
float specularPhongAdv(vec3 V, vec3 L, vec3 N, float sh)
{
	vec3 R = normalize(reflect(V, N));
	float RdotL = max(dot(R, L), 0.0);
	vec3 specv = pow(vec3(RdotL), vec3(sh, sh * 2.0, sh * 8.0));// * vec3(0.333);
	return specv.x + specv.y + specv.z;
}

// Atom method
// a simple heck to simulate the Oren-Nayar effect in a cheap way
// fRoughness in [0, 1]. 0.0 means smooth surface like Lambert
// Require fallOff() heck to smooth shading boundary
float diffuseRough(float NdotL, float fRoughness)
{
	float fR = clamp (fRoughness, 0.0, 1.0);
	float foo = 1.0 - fR;
	return (foo + 1) * 0.5 * pow(NdotL, foo);
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

float truncate(float low, float high, float org)
{
	return min(high, max(low, org));
}

float truncate01(float low, float high, float org)
{
	float r = min(high, max(low, org));
	return (r - low) / (high - low);
}


float cement(vec3 p)
{
	p = p * usp.sizeScale;
	
	if(unp.version == 1){
		float p2 = max(1.0, (perlinNoise3D (p * 2.0) * 0.1 + 0.1) + perlinNoise3D (p * 20.));
		return 1. - pow(0.5 + p2 * 0.5, 30.);
	}else{
		float p2 = max(1.0, (perlinNoise3D (p) * 0.3 + 0.3) + perlinNoise3D (p * 20.));
		return 1. - pow(0.5 + p2 * 0.5, 3.);
	}
}

// main routine
void main()
{
	const float esp = 0.0001;
	// basic vectors
	vec3 vL = normalize(vec3(1.0));
	vec3 vV = normalize(outFragPos);
	vec3 vN = normalize(outNormal);

	// texture lookup
	vec4 kd = vec4(vec3(unp.diffuseColor), 1.0);
	vec4 ks = vec4(vec3(unp.specularColor), 1.0);
	
	// falloff heck
	// This code should be place before the normal being modified by bump mapping
	// SNdorL: surface-normal-dot-lightdirection
	float SNdotL = max(dot(vN, vL), 0.0);
	float fFallOff = fallOff(0.0f, 0.25f, SNdotL);

	float g0 = cement(outPosition);
	float gx = cement(outPosition + vec3(esp, 0.f, 0.f));
	float gy = cement(outPosition + vec3(0.f, esp, 0.f));
	float gz = cement(outPosition + vec3(0.f, 0.f, esp));
	vN = normalize(vN - vec3(gx - g0, gy - g0, gz - g0) * 100.f);

	//kd = kd * g0;
	
	// balance energy
	// balance diffuse and specular energy
	float NdotV = max(dot(-vV, vN), 0.0);
	// select one of the following fresnel method
	float reflFactor = fresnelSchlick(NdotV, unp.reflectivity);
	//float reflFactor = fresnelDielectric(NdotV, 1.f, iIoR);
	//float reflFactor = fresnelConductor (NdotV, eta, extinction)
	float tranFactor = 1.f - reflFactor;
	// do balancing
	ks = ks * reflFactor;
	kd = kd * tranFactor;

	//--- calculate the reflection contribution using an environment map
	// requires "uniform sampler2D u_texReflection[7];"
	// this is for phong-like specular, which is more blur
	// specular reflection
	float shininess = unp.shininess;
	vec4 specRefl = vec4(0.0);
	float fT;
	if (shininess > 0)
	{
		int idx;
		vec3 vR = normalize(reflect(vV, vN));
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
		//vec4 spec0 = texture2D(u_texReflection[idx], vector2ThetaPhi(vR)) * (1.0 - fT);
		//vec4 spec1 = texture2D(u_texReflection[idx+1], vector2ThetaPhi(vR)) * fT;
		
		//specRefl = (spec1 + spec0) * ks;
	}

	// diffuse reflection (ambient)
	//vec4 diffRefl = kd * texture2D(u_texReflection[6], vector2ThetaPhi(vN));
	// total reflection contribution
	//vec4 cReflectinContribution = specRefl + diffRefl;
	//--- end of the reflection contribution computation
	

	float fSpecCoef = specularPhongAdv(vV, vL, vN, unp.shininess);
	float fDiffCoef = diffuseRough(max(dot(vN, vL), 0.0), unp.diffuseRough);

	// shading computtaion
	outColor = gammaCorrection(2.2, (ks * fSpecCoef + kd * fDiffCoef) * fFallOff);
	outColor.a = 1.0;
}