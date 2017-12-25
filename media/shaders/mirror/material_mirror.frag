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
	vec4  glassColor;					// glass color;
	
	float marginSize;
	float tileWidth;
	float tileHeight;
	float albedo;
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
// unit vector to spherical coordinate and scaled to [0, 1] for
// reflection and background texture lookup
vec2 vector2ThetaPhi(vec3 v)
{
	float theta = atan(v.x, v.z);
	float phi = 3.1415926 * 0.5 - acos(v.y);
	return vec2((theta + 3.1415926) * (0.159155), 1.0 - 0.5 * (1.0 + sin(phi)));
}

vec3 calcNormal(vec2 p)
{
	float hMargin = usp.marginSize * 0.5;
	p = p * vec2(usp.tileWidth, usp.tileHeight);
	vec3 vX = vec3(1.0, 0.0, 0.0);
	vec3 vY = vec3(0.0, 1.0, 0.0);
	
	float foox = p.x - usp.tileWidth + hMargin;
	float fooy = p.y - usp.tileHeight + hMargin;
	
	if (p.x < hMargin && p.x < p.y)
	{
	 	vX.z = 1.0;
	 	if (fooy > 0.0 && p.x > hMargin - fooy) vX.z = 0.0;
	}
	if (foox > 0.0 && hMargin - foox < p.y)
	{
		vX.z = -1.0;
		if (fooy > 0.0 && foox < fooy) vX.z = 0.0;
	}
	
	if (p.y < hMargin && p.y < p.x)
	{
		vY.z = 1.0;
		if (foox > 0.0 && p.y > hMargin - foox) vY.z = 0.0;
	}
	if (fooy > 0.0 && hMargin - fooy < p.x)
	{
		vY.z = -1.0;
		if (foox > 0.0 && fooy < foox) vY.z = 0.0;
	}

	return normalize(cross (vX, vY));
}

vec2 fragPos(vec2 coord)
{
	if (coord.x < 0.0) coord.x = coord.x + ceil(abs(coord.x));
	if (coord.y < 0.0) coord.y = coord.y + ceil(abs(coord.y));
	// This ranges from 0.0 to about 3.5. The integer component tells you the row number!

	return fract (coord / vec2(usp.tileWidth, usp.tileHeight));
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
	
	if(unp.version == 2){
		// procedure tiling
		vec2 ppos = fragPos(outTexcoord); // pigment position.
		vec3 vBumpNml = calcNormal(ppos); // pigment normal.
		//vN = (normalize(v_tangent * vBumpNml.x + v_bitangent * vBumpNml.y + vN * vBumpNml.z)).xyz;
	}
 
 	// balance energy
	// balance diffuse and specular energy
	float NdotV = max(dot(-vV, vN), 0.0);
	// select one of the following fresnel method
	//float reflFactor = fresnelDielectric(NdotV, u_IoR, 1.0);
	float reflFactor = fresnelDielectric(NdotV, 1.f, 1.4);
	reflFactor = unp.reflectivity + (1.0 - unp.reflectivity) * reflFactor;
	//float reflFactor = fresnelConductor (NdotV, eta, extinction)
	float tranFactor = 1.f - reflFactor;

	vec3 vR = normalize(reflect(vV, vN));
	//vec4 reflection = texture2D(u_texReflection[0], vector2ThetaPhi(vR));

	// shading computtaion
	outColor = gammaCorrection(2.2, (vec4(reflFactor) + vec4(vec3(usp.glassColor), 1.0) * tranFactor * usp.albedo));
	outColor.a = 1.0;
}