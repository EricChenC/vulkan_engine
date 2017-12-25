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
	vec4  glassColor;
	vec4  glassEdgeColor;
	
	vec4  glassTileColor1;
	vec4  glassTileColor2;
	vec4  glassTileColor3;
	
	float glassThickness;
	float glassEdgeSize;
	
	float glassTilePercentage1;
	float glassTilePercentage2;
	float glassTileSize;
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
float fresnelDielectric(float NdotV, float n1, float n2)
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

vec3 gColor = vec3(0.0);
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
	
	float prob = hash1(bestt);
	if (prob < usp.glassTilePercentage1)
		gColor = vec3(usp.glassTileColor1);
	else if (prob < usp.glassTilePercentage1 + usp.glassTilePercentage2)
		gColor = vec3(usp.glassTileColor2);
	else gColor = vec3(usp.glassTileColor3);

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
	
	// procedure tiling
	vec2 coord = outTexcoord;
	if (coord.x < 0.0) coord.x = coord.x + ceil(abs(coord.x));
	if (coord.y < 0.0) coord.y = coord.y + ceil(abs(coord.y));

	// call the major procedure method
	vec2 bv = voronoi(coord / usp.glassTileSize);

	float h = gDist2Edge;
	vec3 bumpNormal = vec3(0.0);
	if (h >= usp.glassEdgeSize)
	{
		bumpNormal = vec3(0.0, 0.0, 1.0);
		h = usp.glassEdgeSize;
	}
	else
	{
		float a = 3.1415927*.5*h/usp.glassEdgeSize;
		bumpNormal = vec3(-bv*sin(a), cos(a));
		gColor = vec3(usp.glassEdgeColor);
	}
	vec4 kt = vec4(gColor, 1.0);

	// balance energy
	// balance diffuse and specular energy
	float NdotV = max(dot(-vV, vN), 0.0);
	// select one of the following fresnel method
	//float reflFactor = fresnelDielectric(NdotV, unp.indexOfRefraction, 1.0);
	float reflFactor = fresnelDielectric(NdotV, 1.f, unp.indexOfRefraction);
	reflFactor = unp.reflectivity + (1.0 - unp.reflectivity) * reflFactor;
	//float reflFactor = fresnelConductor (NdotV, eta, extinction)
	float tranFactor = 1.f - reflFactor;

	//--- calculate the reflection contribution using an environment map
	// requires "uniform sampler2D u_texReflection[7];"
	// this is for ward-like specular
	// specular reflection
	float shininess = 100.;
	vec3 vR = normalize(reflect(vV, vN));
	//vec4 specRefl = pow(texture2D(u_texReflection[0], vector2ThetaPhi(vR)), vec4(2.2)) * reflFactor;

	vec3 vH = normalize(-vV + vL);

	float NdotL = max(dot(vL, vN), 0.0);
	float NdotH = max(dot(vN, vH), 0.0);
	float HdotXY = sqrt(1.0 - NdotH * NdotH);

	float fSpecCoef = specularWardAdv(NdotL, NdotV, NdotH, HdotXY, 80.);
	
	float fThickness = usp.glassThickness / max(NdotV, 0.01);
	// here we simply use the environment map.
	// In real applications, this sshould be replace by the image behide the glass
	//vec4 vBackground = texture2D(u_texReflection[0], vector2ThetaPhi(vV));
	vec4 vTransmision = kt * vec4(pow(vec3(usp.glassColor), vec3(fThickness)), 1.0) * tranFactor;

	// shading computtaion
	outColor = gammaCorrection(2.2, vec4(reflFactor * fSpecCoef * fFallOff)) + vTransmision;
	outColor.a = 1.0;
}