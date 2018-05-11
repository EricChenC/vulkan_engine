#version 450
#extension GL_ARB_separate_shader_objects : enable

// Output data ; will be interpolated for each fragment.
layout(location = 0) in vec2 UV;
layout(location = 1) in vec3 Position_worldspace;
layout(location = 2) in vec3 Normal_cameraspace;
layout(location = 3) in vec3 EyeDirection_cameraspace;
layout(location = 4) in vec3 LightDirection_cameraspace;
layout(location = 5) in vec4 ShadowCoord;

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


vec2 poissonDisk[16] = vec2[]( 
   vec2( -0.94201624, -0.39906216 ), 
   vec2( 0.94558609, -0.76890725 ), 
   vec2( -0.094184101, -0.92938870 ), 
   vec2( 0.34495938, 0.29387760 ), 
   vec2( -0.91588581, 0.45771432 ), 
   vec2( -0.81544232, -0.87912464 ), 
   vec2( -0.38277543, 0.27676845 ), 
   vec2( 0.97484398, 0.75648379 ), 
   vec2( 0.44323325, -0.97511554 ), 
   vec2( 0.53742981, -0.47373420 ), 
   vec2( -0.26496911, -0.41893023 ), 
   vec2( 0.79197514, 0.19090188 ), 
   vec2( -0.24188840, 0.99706507 ), 
   vec2( -0.81409955, 0.91437590 ), 
   vec2( 0.19984126, 0.78641367 ), 
   vec2( 0.14383161, -0.14100790 ) 
);

// Returns a random number based on a vec3 and an int.
float random(vec3 seed, int i){
	vec4 seed4 = vec4(seed,i);
	float dot_product = dot(seed4, vec4(12.9898,78.233,45.164,94.673));
	return fract(sin(dot_product) * 43758.5453);
}


// main routine
void main()
{

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
    
    float visibility = 1.0;
    float bias = 0.005;
    
    for (int i=0;i<4;i++){
		// use either :
		//  - Always the same samples.
		//    Gives a fixed pattern in the shadow, but no noise
		// int index = i;
		//  - A random sample, based on the pixel's screen location. 
		//    No banding, but the shadow moves with the camera, which looks weird.
		// int index = int(16.0*random(gl_FragCoord.xyy, i))%16;
		//  - A random sample, based on the pixel's position in world space.
		//    The position is rounded to the millimeter to avoid too much aliasing
		int index = int(16.0*random(floor(Position_worldspace.xyz*1000.0), i))%16;
		
		// being fully in the shadow will eat up 4*0.2 = 0.8
		// 0.2 potentially remain, which is quite dark.
        
        float am = 1.0;
        vec4 shc = ShadowCoord / ShadowCoord.w;
        if ( shc.z < 1.0 ) 
        {
            float dist = texture( shadowMap, shc.xy + poissonDisk[index]/700.0 ).r;
            float depth = (ShadowCoord.z - bias) / ShadowCoord.w;
            
            if (dist < depth ) 
            {
                am = 0.5;
            }
        }
        
        
		visibility -= 0.2*(1.0 - am);
	}
    

	vec3 color = 
		// Ambient : simulates indirect lighting
		MaterialAmbientColor +
		// Diffuse : "color" of the object
		visibility * MaterialDiffuseColor * LightColor * LightPower * cosTheta+
		// Specular : reflective highlight, like a mirror
		visibility * MaterialSpecularColor * LightColor * LightPower * pow(cosAlpha,5);
        
    
    outColor = vec4(color, 1.0);
 }