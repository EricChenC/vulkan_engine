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
	vec4  mortarColor;
	vec4  tileColor1;
	vec4  tileColor2;
	vec4  tileColor3;
	vec4  tileSpecularColor;
	
	float mortarSize;
	float tilePercentage1;
	float tilePercentage2;
	float tileSize;
	float tileShininess;
	float tileReflectivity;
	float tileDiffRough;
}usp;

//uniform sampler2D u_texReflection[7];	// the environment map

layout(location = 0) out vec4 outColor;

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

#define HASHSCALE1 .1031
#define HASHSCALE3 vec3(.1031, .1030, .0973)
vec2 hash2(vec2 p)
{
	vec3 p3 = fract(vec3(p.xyx) * HASHSCALE3);
    p3 += dot(p3, p3.yzx+19.19);
    return fract((p3.xx+p3.yz)*p3.zy);

}

float hash1(vec2 p)
{
	vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
    p3 += dot(p3, p3.yzx + 19.19);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 gDiffuse = vec3(0.0);
float gDist2Edge;

vec2 voronoi(vec2 x)
{
	vec2 p1, p2;
	vec2 cp;
	float dist;
	
	vec2 n = floor(x);
	vec2 f = fract(x);

	vec2 mr;

	bool have1 = false;
	int besti, bestj;
	vec2 bestt;
	int lc = 0;

	dist = 1000000.0;
	for( int j = -1; j <= 1; j++ )
	for( int i = -1; i <= 1; i++ )
	{
		vec2 g = vec2(float(i),float(j));
		vec2 t = n + g;
		vec2 tt = t + hash2(t);
		vec2 r = tt - x;
		float newd = length(r);
		vec2 newp = vec2(tt.x, tt.y);

		if(newd < dist)
		{
			dist = newd;
			p1 = newp;
			besti = i;
			bestj = j;
			bestt = t;
		}
	}
	
	float prob;
	if(unp.version == 1){
		prob = hash1(bestt);
		if (prob < usp.tilePercentage1)
			gDiffuse = vec3(usp.tileColor1);
		else if (prob < usp.tilePercentage1 + usp.tilePercentage2)
			gDiffuse = vec3(usp.tileColor2);
		else gDiffuse = vec3(usp.tileColor3);
	}else{
		prob = abs(hash1(bestt));
		gDiffuse = vec3(usp.tileColor1) * prob + vec3(usp.tileColor2) * (1.0 - prob);
	}

	vec2 r2;
	vec2 bv;
	
	gDist2Edge = 1000000.0;
	for( int j = -2; j <= 2; j++ )
	for( int i = -2; i <= 2; i++ )
 	{
		if(i==besti && j==bestj) continue;
		vec2 g = vec2(float(i),float(j));
		vec2 t = n + g;
		t += hash2(t);
		vec2 newp = t;
 
		vec2 c2 = .5 * (p1.xy + newp.xy);
 		float newd;
 
		r2 = normalize(p1.xy - c2);
		newd = dot(r2, x - c2);
 
 		if(newd < gDist2Edge)
 		{
 			gDist2Edge = newd;
 			p2 = newp;
			cp = c2;
			bv = r2;
 		}
 	}
 	
	return bv;
}

void main( void )
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
  
   	vec4 ks = vec4(vec3(usp.tileSpecularColor), 1.0);
	float shininess = usp.tileShininess;
	
	// procedure tiling
	vec2 coord = outTexcoord;
	if (coord.x < 0.0) coord.x = coord.x + ceil(abs(coord.x));
	if (coord.y < 0.0) coord.y = coord.y + ceil(abs(coord.y));

	// call the major procedure method
	vec2 bv = voronoi(coord / usp.tileSize);

	float h = gDist2Edge;
	vec3 bumpNormal = vec3(0.0);
	if (h >= usp.mortarSize)
	{
		bumpNormal = vec3(0.0, 0.0, 1.0);
		h = usp.mortarSize;
	}
	else
	{
		float a = 3.1415927*.5*h/usp.mortarSize;
		bumpNormal = vec3(-bv*sin(a), cos(a));
		gDiffuse = vec3(usp.mortarColor);
		
		if(unp.version !=2){
			ks = vec4(0.0);
		}
		
		shininess = 0.0;
	}
	vec4 kd = vec4(gDiffuse, 1.0);
	// mapping to world coordinate
   	//vN = (normalize(v_tangent * bumpNormal.x + v_bitangent * bumpNormal.y + vN * bumpNormal.z)).xyz;

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
	
	float fSpecCoef = specularPhongAdv(vV, vL, vN, usp.tileShininess);
	float fDiffCoef = diffuseRough(max(dot(vN, vL), 0.0), usp.tileDiffRough);
   
	outColor = gammaCorrection(2.2, (ks * fSpecCoef + kd * fDiffCoef) * fFallOff);
	outColor.a = 1.0;
}