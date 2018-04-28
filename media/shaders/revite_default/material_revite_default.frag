#version 450
#extension GL_ARB_separate_shader_objects : enable

// wood texture
// data from vertex shader
layout(location = 1) in vec2 outTexcoord;
layout(location = 2) in vec3 outNormal;
layout(location = 3) in vec3 outLightPos;
layout(location = 4) in vec3 outFragPos;	// out view
layout(location = 5) in vec3 outPosition;
layout(location = 6) in vec4 outShadowCoord;

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

layout(binding = 3) uniform sampler2D diffuseTexture;
layout(binding = 4) uniform sampler2D specularTexture;
layout(binding = 5) uniform sampler2D bumpTexture;

// special
layout(binding = 6) uniform UniformSpecialParameters{
	vec4 tintColor;	

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
layout(binding = 9) uniform sampler2D shadowMap;

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

mat3 cotangent_frame(vec3 N, vec3 p, vec2 uv)
{
    // get edge vectors of the pixel triangle
    vec3 dp1 = dFdx( p );
    vec3 dp2 = dFdy( p );
    vec2 duv1 = dFdx( uv );
    vec2 duv2 = dFdy( uv );
 
    // solve the linear system
    vec3 dp2perp = cross( dp2, N );
    vec3 dp1perp = cross( N, dp1 );
    vec3 T = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 B = dp2perp * duv1.y + dp1perp * duv2.y;
 
    // construct a scale-invariant frame 
    float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
    return mat3( T * invmax, B * invmax, N );
}

vec3 perturb_normal( vec3 N, vec3 V, vec2 texcoord )
{
    // assume N, the interpolated vertex normal and 
    // V, the view vector (vertex to eye)
   vec3 map = texture(bumpTexture, texcoord ).xyz;
   map = map  - 64./127.5;
    mat3 TBN = cotangent_frame(N, -V, texcoord);
    return normalize(TBN * map);
}

#define ambient 0.1

float textureProj(vec4 P, vec2 off)
{
	float shadow = 1.0;
	vec4 shadowCoord = P / P.w;
	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
	{
		float dist = texture( shadowMap, shadowCoord.st + off ).r;
		if ( shadowCoord.w > 0.0 && dist < shadowCoord.z ) 
		{
			shadow = ambient;
		}
	}
	return shadow;
}

float filterPCF(vec4 sc)
{
	ivec2 texDim = textureSize(shadowMap, 0);
	float scale = 1.5;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;
	
	for (int x = -range; x <= range; x++)
	{
		for (int y = -range; y <= range; y++)
		{
			shadowFactor += textureProj(sc, vec2(dx*x, dy*y));
			count++;
		}
	
	}
	return shadowFactor / count;
}

// main routine
void main()
{
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
	
	vec2 texCutOffCoord = outTexcoord * ustp.cutOffRepeat + ustp.cutOffOffset;
	float fAlpha = luminance(texture(cutOffTexture, texCutOffCoord).xyz);
	// if (fAlpha > usp.cutOff) discard;
	
	// diffuse texture (intrinsic color)
	vec2 texDiffCoord = outTexcoord * untp.diffuseRepeat + untp.diffuseOffset;
	kd = kd * texture(diffuseTexture, texDiffCoord);
	
	// specular texture (specular reflectivity) 
	vec2 texSpecCoord = outTexcoord * untp.specularRepeat + untp.specularOffset;
	ks = ks * texture(specularTexture, texSpecCoord);
	
	// specular shininess texture 
	//vec2 texShinCoord = outTexcoord * u_texShininess.repeat+ u_texShininess.offset;
	//float shininess = unp.shininess * luminance(texture(u_texShininess.bitmap, texShinCoord).xyz) * 99.0;
	
	// bump texture
	vec2 texBumpCoord = outTexcoord * untp.bumpRepeat + untp.bumpOffset;
	//vec3 vBumpNml = color2Normal(texture(bumpTexture, texBumpCoord).xyz);
	vec3 vBumpNml = perturb_normal(outNormal, outFragPos, texBumpCoord);
	
	//vBumpNml = (normalize(v_tangent * vBumpNml.x + v_bitangent * vBumpNml.y + vN * vBumpNml.z)).xyz;
	vN = normalize(vBumpNml * untp.bumpScale + vN * (1.0 - untp.bumpScale));
	
	// balance energy
	// balance diffuse and specular energy
	float NdotV = max(dot(-vV, vN), 0.0);
	// select one of the following fresnel method
	//float reflFactor = fresnelDielectric(NdotV, unp.indexOfRefraction, 1.0);
	float reflFactor = fresnelDielectric(NdotV, 1.f, unp.indexOfRefraction);
	reflFactor = unp.reflectivity + (1.0 - unp.reflectivity) * reflFactor;
	//float reflFactor = fresnelConductor (NdotV, eta, extinction)
	float tranFactor = 1.f - reflFactor;
	// do balancing
	ks = ks * reflFactor;
	kd = kd * tranFactor;

	//--- calculate the reflection contribution using an environment map
	// requires "uniform sampler2D u_texReflection[7];"
	// this is for ward-like specular
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
		vec4 spec0 = (usp.tintColor + unp.ambientColor) * (1.0 - fT);
		vec4 spec1 = (usp.tintColor + unp.ambientColor) * fT;
		
		specRefl = (spec1 + spec0) * ks;
	}

	// diffuse reflection (ambient)
	vec4 diffRefl = kd * (usp.tintColor + unp.ambientColor);
	// total reflection contribution
	vec4 cReflectinContribution = specRefl + diffRefl;
	//--- end of the reflection contribution computation
	

	vec3 vH = normalize(-vV + vL);

	float NdotL = max(dot(vL, vN), 0.0);
	float NdotH = max(dot(vN, vH), 0.0);
	float HdotXY = sqrt(1.0 - NdotH * NdotH);
    
    float shadow = textureProj(outShadowCoord / outShadowCoord.w, vec2(0.0));

	float fSpecCoef = specularWardAdv(NdotL, NdotV, NdotH, HdotXY, unp.shininess);
	float fDiffCoef = diffuseOrenNayar(unp.diffuseRough, vN, vL, vV, NdotL, NdotV);

	// shading computtaion
	outColor = gammaCorrection(2.2, 
		(ks * fSpecCoef + kd * fDiffCoef * shadow) * fFallOff + cReflectinContribution);
	outColor.a = 1.0;
}