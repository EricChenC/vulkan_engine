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

layout(binding = 2) uniform UniformNormalTextureParameters{
	vec2 		diffuseOffset;		// UV pixel offset
	vec2 		diffuseRepeat;		// UV pixel repeat
	
	vec2 		specularOffset;		// UV pixel offset
	vec2 		specularRepeat;		// UV pixel repeat
	
	vec2 		bumpOffset;			// UV pixel offset
	vec2 		bumpRepeat;			// UV pixel repeat
	
	float		diffuseScale;		// diffuse scale
	float		specularScale;		// specular scale
	float		bumpScale;			// bump scale
}untp;

layout(binding = 5) uniform sampler2D bumpTexture;

// special
layout(binding = 6) uniform UniformSpecialParameters{
	float shininessU;				// specular shininess
	float shininessV;				// specular shininess
	float cutOff;					// cut-off opacity
}usp;

layout(binding = 7) uniform UniformSpecialTextureParameters{
	vec2 		cutOffOffset;		// UV pixel offset
	vec2 		cutOffRepeat;		// UV pixel repeat
	float		cutOffScale;		// default scale
}ustp;


layout(binding = 8) uniform sampler2D cutOffTexture;

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
	
	if(unp.version == 1){
		// cutoff texture (specular reflectivity) 
		vec2 texCutOffCoord = outTexcoord * ustp.cutOffRepeat + ustp.cutOffOffset;
		float fAlpha = luminance(texture(cutOffTexture, texCutOffCoord).xyz);
		if (fAlpha < usp.cutOff) discard;
	}
	
	// bump texture
	vec2 texBumpCoord = outTexcoord * untp.bumpRepeat + untp.bumpOffset;
	vec3 vBumpNml = color2Normal(texture(bumpTexture, texBumpCoord).xyz);
	//vBumpNml = (normalize(v_tangent * vBumpNml.x + v_bitangent * vBumpNml.y + vN * vBumpNml.z)).xyz;
	vN = normalize(vBumpNml * untp.bumpScale + vN * (1.0 - untp.bumpScale));

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
	// if (usp.shininessU == usp.shininessV)
	// {
		float HdotXY = sqrt(1.0 - NdotH * NdotH);
		fSpecCoef = specularWardAdv(NdotL, NdotV, NdotH, HdotXY, usp.shininessU);
	// }
	// else
	// {
		// float HdotX = dot(vH, v_tangent);
		// float HdotY = dot(vH, v_bitangent);
		// fSpecCoef = specularWard(NdotL, NdotV, NdotH, HdotX, HdotY, usp.shininessU, usp.shininessV);
	// }

	//--- calculate the reflection contribution using an environment map
	// requires "uniform sampler2D u_texReflection[7];"
	// this is for ward-like specular
	// specular reflection
	float shininess = (usp.shininessU + usp.shininessV) * 0.5;
	vec4 specRefl = vec4(0.4);
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
		//vec4 spec0 = texture(u_texReflection[idx-1], vector2ThetaPhi(vR)) * (1.0 - fT);
		//vec4 spec1 = texture(u_texReflection[idx], vector2ThetaPhi(vR)) * fT;
		
		//specRefl = (spec1 + spec0) * ks;
	}
	//--- end of the reflection contribution computation
	
	// shading computtaion
	outColor = gammaCorrection(2.2, ks * fSpecCoef * fFallOff + specRefl);
}