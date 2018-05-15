#version 450
#extension GL_ARB_separate_shader_objects : enable

// matrices
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec3 lightDir;
} ubo;

layout(push_constant) uniform ModelMatrix {
    mat4 model;
} mt;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;


// Output data ; will be interpolated for each fragment.
layout (location = 0) out vec2 UV;
layout (location = 1) out vec3 Position_worldspace;
layout (location = 2) out vec3 Normal_cameraspace;
layout (location = 3) out vec3 EyeDirection_cameraspace;
layout (location = 4) out vec3 LightDirection_cameraspace;
layout (location = 5) out vec3 outViewPos;



out gl_PerVertex {
    vec4 gl_Position;
};

// vertex shader
void main()
{
		// Output position of the vertex, in clip space : MVP * position
	gl_Position =  ubo.proj * ubo.view * mt.model * vec4(inPosition,1);
	
	
	// Position of the vertex, in worldspace : M * position
	Position_worldspace = (mt.model * vec4(inPosition,1)).xyz;
	
	// Vector that goes from the vertex to the camera, in camera space.
	// In camera space, the camera is at the origin (0,0,0).
	EyeDirection_cameraspace = vec3(0,0,0) - ( ubo.view * mt.model * vec4(inPosition,1)).xyz;

	// Vector that goes from the vertex to the light, in camera space
	LightDirection_cameraspace = (ubo.view * vec4(ubo.lightDir,0)).xyz;
	
	// Normal of the the vertex, in camera space
	Normal_cameraspace = ( ubo.view * mt.model * vec4(inNormal,0)).xyz; // Only correct if ModelMatrix does not scale the model ! Use its inverse transpose if not.
	
	// UV of the vertex. No special space for this one.
	UV = inTexcoord;
    
    outViewPos = (ubo.view * mt.model * vec4(inPosition.xyz, 1.0)).xyz;
}