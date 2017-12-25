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