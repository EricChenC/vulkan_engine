#version 450
#extension GL_ARB_separate_shader_objects : enable

// wood proc
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

layout(binding = 3) uniform sampler2D diffuseTexture;
layout(binding = 4) uniform sampler2D specularTexture;
layout(binding = 5) uniform sampler2D bumpTexture;

// special
layout(binding = 6) uniform UniformSpecialParameters{
	vec4 lineColor;
	vec4 baseColor;
	vec4 gapColor;

	float knobFactor;
	float width;
	float heightFactor;
	
	float slantFactor;
	float cycleDensity;
	float cycleHardness;
	float oddOffset;
}usp;

layout(binding = 7) uniform UniformSpecialTextureParameters{
	vec2 		defaultOffset;		// UV pixel offset
	vec2 		defaultRepeat;		// UV pixel repeat
	float		defaultScale;		// default scale
}ustp;

layout(binding = 8) uniform sampler2D defaultTexture;

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

float hash(vec2 n)
{ 
    return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

float noise(vec2 p)
{
    vec2 ip = floor(p);
    vec2 u = fract(p);
    u = u*u*(3.0-2.0*u);

    float res = mix(
        mix(hash(ip),hash(ip+vec2(1.0,0.0)),u.x),
        mix(hash(ip+vec2(0.0,1.0)),hash(ip+vec2(1.0,1.0)),u.x),u.y);
    return res;
}

float height(vec2 a)
{
    a = a*vec2(1.0, 1.0/max(usp.slantFactor, 0.1));
    return usp.cycleDensity * (usp.knobFactor*noise(a) + a.x);
}

vec2 grad(in vec2 x)
{
	vec2 h = vec2(0.05, 0.0);
	return vec2(height(x+h.xy) - height(x-h.xy),
                height(x+h.yx) - height(x-h.yx))/(2.0*h.x);
}


vec4 woodProc(vec2 uv)
{
    vec2  pos = uv / usp.width;
    float plank = floor(pos.x); // unique per plank
    float offset = (fract(plank * 0.5) > 0.3) ? usp.oddOffset : 0.0;
	float py = pos.y/usp.heightFactor+offset;
    float item = floor(py);
    
    vec2  pixel = pos + vec2(124., 11.) * plank;
    float value = height(pixel + item);
    vec2  gradient = grad(pixel + item);
    float linePos = 1.0 - smoothstep(0.0, 0.08, fract(value)/length(gradient));
    float line = floor(value); // unique per line
    float lineWeight = mix(1.0, 0.4+hash(vec2(line,plank)), 0.8);
    float lineGrain = smoothstep(-0.2, 0.9, 0.6);
    
    float darkness = mix(1.0, 0.5+hash(vec2(plank, item)), 0.2);
    float grain = mix(1.0, 0.9, 0.1);
    
    float gapY = step(0.0, fract(pos.x)) * (1.0-step(0.02, fract(pos.x)));
    float gapX = step(0.0, fract(py)) * (1.0-step(0.02/usp.heightFactor, fract(py)));

	float fGap = max(gapY, gapX);
	float fLine = usp.cycleHardness*lineWeight*lineGrain*linePos;
	
	float fSpec = min((1. - fLine) * 0.3 + 0.7, 1 - fGap);
	return vec4(mix(mix(usp.baseColor.xyz, usp.lineColor.xyz, fLine), usp.gapColor.xyz, fGap), fSpec);
}

// main routine
void main()
{
	// basic vectors
	vec3 vL = normalize(vec3(1.0));
	vec3 vV = normalize(outFragPos);
	vec3 vN = normalize(outNormal);

	// falloff heck
	// This code should be place before the normal being modified by bump mapping
	// SNdorL: surface-normal-dot-lightdirection
	float SNdotL = max(dot(vN, vL), 0.0);
	float fFallOff = fallOff(0.0f, 0.25f, SNdotL);

	vec4 wood = woodProc(outTexcoord);
	vec4 kd = vec4(unp.diffuseColor.xyz * wood.xyz, 1.0);
	vec4 ks = vec4(unp.specularColor.xyz * wood.w, 1.0);
	
	// balance energy
	// balance diffuse and specular energy
	float NdotV = max(dot(-vV, vN), 0.0);
	// select one of the following fresnel method
	//float reflFactor = fresnelDielectric(NdotV, indexOfRefraction, 1.0);
	float reflFactor = fresnelDielectric(NdotV, 1.f, unp.indexOfRefraction);
	reflFactor = unp.reflectivity + (1.0 - unp.reflectivity) * reflFactor;
	//float reflFactor = fresnelConductor (NdotV, eta, extinction)
	float tranFactor = 1.f - reflFactor;
	// do balancing
	ks = ks * reflFactor;
	kd = kd * tranFactor;

	vec3 vH = normalize(-vV + vL);

	float NdotL = max(dot(vL, vN), 0.0);
	float NdotH = max(dot(vN, vH), 0.0);
	float HdotXY = sqrt(1.0 - NdotH * NdotH);

	float fSpecCoef = specularWardAdv(NdotL, NdotV, NdotH, HdotXY, unp.shininess);
	float fDiffCoef = diffuseOrenNayar(unp.diffuseRough, vN, vL, vV, NdotL, NdotV);

	// shading computtaion
	outColor = gammaCorrection(2.2, 
		(ks * fSpecCoef + kd * fDiffCoef) * fFallOff);
	outColor.a = 1.0;
}