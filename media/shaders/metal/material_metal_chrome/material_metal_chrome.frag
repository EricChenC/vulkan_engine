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

// special
layout(binding = 6) uniform UniformSpecialParameters{
	float metalShininess;				// specular shininess
	float metalContrast;
	float metalSaturation;
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
// for Schlick Fresnel computation
float pow5(float x)
{
	float foo = x * x;
	return foo * foo * x;
}

// Atom method
// Precise Fresnel computation for conductor surface
// Require pow5() method
float fresnelConductor(float NdotV, float ior, float extinction)
{
	float cosi = clamp(NdotV, -1.0, 1.0);

	float numerator1 = (ior - 1.0) * (ior - 1.0);
	float denominator = (ior + 1.0) * (ior + 1.0);
	float numerator2 = 4.0 * ior * pow5(1.0 - cosi);
	float k2 = extinction * extinction;
	return (numerator1 + numerator2 + k2) / (denominator + k2);
}

// Atom method
// specular computation using the Ward model
float specularWard(float NdotL, float NdotV, float NdotH, float HdotX, float HdotY, float shinyX, float shinyY)
{
	shinyX = shinyX / 10.0;
	shinyY = shinyY / 10.0;
	if (NdotL * NdotV < 0.0001) return 0.0;

	float su2 = shinyX * shinyX;
	float sv2 = shinyY * shinyY;
	float suv = shinyX * shinyY;

	float NH2 = NdotH * NdotH;
	float HU2 = HdotX * HdotX;
	float HV2 = HdotY * HdotY;

	float plus = HU2 * su2 + HV2 * sv2;
	if (NH2 < 0.0001 * plus)
	{
		return 0.0;
	}
	else
	{
		float expo = -plus / NH2;
		return (exp(expo) * suv) / (sqrt(NdotL * NdotV) * 12.56637);
	}
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

// Atom method
// Adjust contrast, a heck for metal reflection
vec4 adjustContrast(const vec4 rgb)
{
	vec3 rsl = (rgb.xyz - vec3(0.5)) * usp.metalContrast + vec3(0.5);
	if (rsl.x < 0.0) rsl.x = 0.0;
	if (rsl.y < 0.0) rsl.y = 0.0;
	if (rsl.z < 0.0) rsl.z = 0.0;
	return vec4(rsl, 1.0);
}

// Atom method
// RGB to HSL
vec3 RGBtoHSL(const vec3 rgb)
{
	vec3 hsl;

	float h = 0.0;
	float s = 0.0;
	float l = 0.0;
	// normalizes red-green-blue values  
	float maxVal = max(max(rgb.x, rgb.y), rgb.z);
	float minVal = min(min(rgb.x, rgb.y), rgb.z);

	// hue  
	if (maxVal == minVal)						h = 0; // undefined  
	else if (maxVal == rgb.x && rgb.y >= rgb.z)	h = 60.0*(rgb.y - rgb.z) / (maxVal - minVal);
	else if (maxVal == rgb.x && rgb.y < rgb.z)	h = 60.0*(rgb.y - rgb.z) / (maxVal - minVal) + 360.0;
	else if (maxVal == rgb.y)					h = 60.0*(rgb.z - rgb.x) / (maxVal - minVal) + 120.0;
	else if (maxVal == rgb.z)					h = 60.0*(rgb.x - rgb.y) / (maxVal - minVal) + 240.0;

	// luminance  
	l = (maxVal + minVal) / 2.0;

	// saturation  
	if (l == 0 || maxVal == minVal)	s = 0;
	else if (0 < l && l <= 0.5)		s = (maxVal - minVal) / (maxVal + minVal);
	else if (l > 0.5)				s = (maxVal - minVal) / (2 - (maxVal + minVal));

	hsl.x = (h>360.) ? 360. : ((h<0.) ? 0. : h);
	hsl.y = ((s>1.) ? 1. : ((s<0.) ? 0. : s)) * 100.;
	hsl.z = ((l>1.) ? 1. : ((l<0.) ? 0. : l)) * 100.;

	return hsl;
}

// Atom method
// Converts HSL to RGB  
vec3 HSLtoRGB(const vec3 hsl)
{
	vec3 rgb;

	float h = hsl.x;		// h must be [0, 360]  
	float s = hsl.y / 100.;	// s must be [0, 1]  
	float l = hsl.z / 100.;	// l must be [0, 1] 

	//float R, G, B;
	if (hsl.y == 0)
	{
		// achromatic color (gray scale)
		rgb = vec3(l);
	}
	else
	{
		float q = (l<0.5) ? (l * (1.0 + s)) : (l + s - (l*s));
		float p = (2.0 * l) - q;
		float Hk = h / 360.0;
		float T[3];
		T[0] = Hk + 0.3333333; // Tr   0.3333333f=1.0/3.0  
		T[1] = Hk;             // Tb  
		T[2] = Hk - 0.3333333; // Tg  
		for (int i = 0; i<3; i++)
		{
			if (T[i] < 0)	T[i] += 1.0;
			if (T[i] > 1)	T[i] -= 1.0;

			if ((T[i] * 6) < 1)			T[i] = p + ((q - p)*6.0*T[i]);
			else if ((T[i] * 2.0) < 1)	T[i] = q;
			else if ((T[i] * 3.0) < 2)	T[i] = p + (q - p) * ((2.0 / 3.0) - T[i]) * 6.0;
			else						T[i] = p;
		}
		rgb.x = max(0., T[0]);
		rgb.y = max(0., T[1]);
		rgb.z = max(0., T[2]);
	}

	return rgb;
}

// Atom method
// Adjust saturation  
vec4 adjustSaturation(vec4 col)
{
	col.xyz = RGBtoHSL(col.xyz);
	col.y *= usp.metalSaturation;
	return vec4(HSLtoRGB(col.xyz), 1.0);
}


// main routine
void main()
{
	// basic vectors
	vec3 vL = normalize(vec3(1.0));
	vec3 vV = normalize(outFragPos);
	vec3 vN = normalize(outNormal);

	// texture lookup
	vec4 ks = vec4(vec3(unp.specularColor), 1.0);
	
	// falloff heck
	// This code should be place before the normal being modified by bump mapping
	// SNdorL: surface-normal-dot-lightdirection
	float SNdotL = max(dot(vN, vL), 0.0);
	float fFallOff = fallOff(0.0f, 0.25f, SNdotL);
	
	// balance energy
	// balance diffuse and specular energy
	float NdotV = max(dot(-vV, vN), 0.0);
	// select one of the following fresnel method
	//float reflFactor = fresnelDielectric(NdotV, unp.indexOfRefraction, 1.0);
	float reflFactor = fresnelConductor(NdotV, unp.indexOfRefraction, unp.extinction);
	reflFactor = unp.reflectivity + (1.0 - unp.reflectivity) * reflFactor;
	ks = ks * reflFactor;

	vec3 vH = normalize(-vV + vL);

	float NdotL = max(dot(vL, vN), 0.0);
	float NdotH = max(dot(vN, vH), 0.0);

	float fSpecCoef;

	float HdotXY = sqrt(1.0 - NdotH * NdotH);
	fSpecCoef = specularWardAdv(NdotL, NdotV, NdotH, HdotXY, usp.metalShininess);

	//--- calculate the reflection contribution using an environment map
	// requires "uniform sampler2D u_texReflection[7];"
	// this is for ward-like specular
	// specular reflection
	float shininess = usp.metalShininess;
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
		//vec4 spec0 = texture2D(u_texReflection[idx-1], vector2ThetaPhi(vR)) * (1.0 - fT);
		//vec4 spec1 = texture2D(u_texReflection[idx], vector2ThetaPhi(vR)) * fT;
		
		//specRefl = (spec1 + spec0) * ks;
	}
	//--- end of the reflection contribution computation
	specRefl = adjustContrast(adjustSaturation(specRefl));
	// shading computtaion
	outColor = gammaCorrection(2.2, ks * fSpecCoef * fFallOff + specRefl);
}