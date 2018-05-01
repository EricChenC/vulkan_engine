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

	vec4 kd = texture(diffuseTexture, outTexcoord);
    
    // float shadow = textureProj(outShadowCoord / outShadowCoord.w, vec2(0.0));
    float shadow = filterPCF(outShadowCoord / outShadowCoord.w);
    
    shadow = max(shadow, 0.4);
    
    vec3 N = normalize(outNormal);
	vec3 L = normalize(outLightPos);
	vec3 V = normalize(outFragPos);
	vec3 R = normalize(-reflect(L, N));
	vec3 diffuse = max(dot(N, L), ambient) * vec3(1.0);
    
	// shading computtaion
	outColor = vec4(diffuse * shadow, 1.0);
}