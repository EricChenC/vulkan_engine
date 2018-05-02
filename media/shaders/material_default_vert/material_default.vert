#version 450
#extension GL_ARB_separate_shader_objects : enable

// matrices
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    mat4 lightSpace;
	vec3 lightPos;
} ubo;

layout(push_constant) uniform ModelMatrix {
    mat4 model;
} mt;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;

layout(location = 1) out vec2 outTexcoord;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outLightPos;
layout(location = 4) out vec3 outFragPos;	// out view
layout(location = 5) out vec3 outPosition;
layout(location = 6) out vec4 outShadowCoord;

out gl_PerVertex {
    vec4 gl_Position;
};

// vertex shader
void main()
{
	// Note: it is more efficient to calculate mvp_matrix outside

	// calculate world(camera) position
	vec4 pos = mt.model * vec4(inPosition, 1.0);

	// project to screen
	gl_Position = ubo.proj * ubo.view * mt.model * vec4(inPosition, 1.0);

	// output to fragment shader
	outTexcoord = inTexcoord;
	outNormal = mat3(mt.model) * inNormal;
	outLightPos = normalize(ubo.lightPos - inPosition);
	outFragPos = -pos.xyz;
	outPosition = inPosition;

	outShadowCoord = ( ubo.lightSpace * mt.model ) * vec4(inPosition, 1.0);
}