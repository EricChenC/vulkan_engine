#version 450
#extension GL_ARB_separate_shader_objects : enable


#define SHADOW_MAP_CASCADE_COUNT 4

layout (location = 0) in vec2 UV;
layout (location = 1) in vec3 Position_worldspace;
layout (location = 2) in vec3 Normal_cameraspace;
layout (location = 3) in vec3 EyeDirection_cameraspace;
layout (location = 4) in vec3 LightDirection_cameraspace;
layout (location = 5) in vec3 inViewPos;


layout(binding = 1) uniform sampler2DArray shadowMap;

layout (binding = 2) uniform CSM {
	vec4 cascadeSplits;
	mat4 cascadeViewProjMat[SHADOW_MAP_CASCADE_COUNT];
} csm;

layout(location = 0) out vec4 outFragColor;


const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 
);


#define ambient 0.3

float textureProj(vec4 P, vec2 offset, uint cascadeIndex)
{
	float shadow = 1.0;
	float bias = 0.005;

	vec4 shadowCoord = P / P.w;
	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) {
		float dist = texture(shadowMap, vec3(shadowCoord.st + offset, cascadeIndex)).r;
		if (shadowCoord.w > 0 && dist < shadowCoord.z - bias) {
			shadow = ambient;
		}
	}
	return shadow;

}


float filterPCF(vec4 sc, uint cascadeIndex)
{
	ivec2 texDim = textureSize(shadowMap, 0).xy;
	float scale = 0.75;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;
	
	for (int x = -range; x <= range; x++) {
		for (int y = -range; y <= range; y++) {
			shadowFactor += textureProj(sc, vec2(dx*x, dy*y), cascadeIndex);
			count++;
		}
	}
	return shadowFactor / count;
}



// main routine
void main()
{


	// Get cascade index for the current fragment's view position
	uint cascadeIndex = 0;
	for(uint i = 0; i < SHADOW_MAP_CASCADE_COUNT - 1; ++i) {
		if(inViewPos.z < csm.cascadeSplits[i]) {
			cascadeIndex = i + 1;
		}
	}

	// Depth compare for shadowing
	vec4 shadowCoord = (biasMat * csm.cascadeViewProjMat[cascadeIndex]) * vec4(Position_worldspace, 1.0);	
    
    
    // Light emission properties
	vec3 LightColor = vec3(1,1,1);
	float LightPower = 1.0f;
	
	// Material properties
	vec3 MaterialDiffuseColor = vec3(1.0, 1.0, 1.0);
	vec3 MaterialAmbientColor = vec3(0.1,0.1,0.1) * MaterialDiffuseColor;
	vec3 MaterialSpecularColor = vec3(0.3,0.3,0.3);

	// Distance to the light
	//float distance = length( LightPosition_worldspace - Position_worldspace );

	// Normal of the computed fragment, in camera space
	vec3 n = normalize( Normal_cameraspace );
	// Direction of the light (from the fragment to the light)
	vec3 l = normalize( LightDirection_cameraspace );
	// Cosine of the angle between the normal and the light direction, 
	// clamped above 0
	//  - light is at the vertical of the triangle -> 1
	//  - light is perpendiular to the triangle -> 0
	//  - light is behind the triangle -> 0
	float cosTheta = clamp( dot( n,l ), 0,1 );
	
	// Eye vector (towards the camera)
	vec3 E = normalize(EyeDirection_cameraspace);
	// Direction in which the triangle reflects the light
	vec3 R = reflect(-l,n);
	// Cosine of the angle between the Eye vector and the Reflect vector,
	// clamped to 0
	//  - Looking into the reflection -> 1
	//  - Looking elsewhere -> < 1
	float cosAlpha = clamp( dot( E,R ), 0,1 );
	
	float visibility=1.0;

        
    float shadow = 1.0;
    shadow = filterPCF(shadowCoord / shadowCoord.w, cascadeIndex);
    
    visibility -= 0.8*(1.0 - shadow);
        
    
	
	vec3 color = 
		// Ambient : simulates indirect lighting
		MaterialAmbientColor +
		// Diffuse : "color" of the object
		visibility * MaterialDiffuseColor * LightColor * LightPower * cosTheta+
		// Specular : reflective highlight, like a mirror
		visibility * MaterialSpecularColor * LightColor * LightPower * pow(cosAlpha,5);
    
    
    outFragColor = vec4(color, 1.0f);
    
    
    
 }