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
	vec4 marbleColor01;
	vec4 marbleColor02;
	vec4 marbleColor11;
	vec4 marbleColor12;
	vec4 gapColor;
	vec4 gapSpecular;

	float sizeScale;
	float tileWidth;
	float tileHeight;
	float tileThick;
	float gapWidth;
	float marbleShininess;			// specular shininess
	float gapShininess;				
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


// methods for procedure marble texture
float hash(float x, float y)
{
	vec3 p3  = fract(vec3(x, y, x) * .1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float hash(float x, float y, float z)
{
	vec3 p3 = vec3(x, y, z);
	p3  = fract(p3 * .1031);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

float perlin(vec3 uvw)
{
	float col = 0.;
	for (int i = 0; i < 8; i ++)
	{
    	vec3 f = floor(uvw), c = f+1.;
    	col += mix(mix (mix(hash(f.x, f.y, f.z), hash(f.x, f.y, c.z), fract(uvw.z)),
                	    mix(hash(f.x, c.y, f.z), hash(f.x, c.y, c.z), fract(uvw.z)), fract(uvw.y)),
        	       mix (mix(hash(c.x, f.y, f.z), hash(c.x, f.y, c.z), fract(uvw.z)),
            	        mix(hash(c.x, c.y, f.z), hash(c.x, c.y, c.z), fract(uvw.z)), fract(uvw.y)), fract(uvw.x));


		col *= .5;
		uvw *= .5;
	}
	
	return col;
}

// main routine
void main()
{
	// basic vectors
	vec3 vL = normalize(vec3(1.0));
	vec3 vV = normalize(outFragPos);
	vec3 vN = normalize(outNormal);
	
	// do tiling
	vec3 tilePos = outPosition / vec3(usp.tileWidth, usp.tileHeight, usp.tileThick);
	vec3 grid = floor (tilePos);
	tilePos = tilePos - grid;

	// procedure marble
	vec3 p = (outPosition + grid)* 800.0 / usp.sizeScale;
	vec3 uv = p + perlin(p)*600.;
	float t = perlin(p + perlin(uv)*400.);
	t = t * t * t;
	
	vec4 kd = (fract((grid.x + grid.y + grid.z) * 0.5) > 0.3) ?
		vec4(vec3(usp.marbleColor01) * t + vec3(usp.marbleColor02) * (1.0 - t), 1.0) :
		vec4(vec3(usp.marbleColor11) * t + vec3(usp.marbleColor12) * (1.0 - t), 1.0);

	vec4 ks = vec4(vec3(unp.specularColor), 1.0);
	float shininess = usp.marbleShininess;
	if (tilePos.x < usp.gapWidth || tilePos.y < usp.gapWidth || tilePos.z < usp.gapWidth)
	{
		kd = vec4(vec3(usp.gapColor), 1.0);
		ks = vec4(vec3(usp.gapSpecular), 1.0);
		shininess = usp.gapShininess;
	}
	
	// falloff heck
	// This code should be place before the normal being modified by bump mapping
	// SNdorL: surface-normal-dot-lightdirection
	float SNdotL = max(dot(vN, vL), 0.0);
	float fFallOff = fallOff(0.0f, 0.25f, SNdotL);
	
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

	float fSpecCoef = specularPhongAdv(vV, vL, vN, shininess);
	float fDiffCoef = diffuseRough(max(dot(vN, vL), 0.0), unp.diffuseRough);

	// shading computtaion
	outColor = gammaCorrection(2.2, (ks * fSpecCoef + kd * fDiffCoef) * fFallOff );
	outColor.a = 1.0;
}