#version 450
#extension GL_ARB_separate_shader_objects : enable

// matrices
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform ModelMatrix {
    mat4 model;
} mt;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;

layout(location = 1) out smooth vec2 outTexcoord;
layout(location = 2) out smooth vec3 outNormal;
layout(location = 3) out smooth vec3 outLightPos;
layout(location = 4) out smooth vec3 outFragPos;	// out view
layout(location = 5) out smooth vec3 outPosition;

out gl_PerVertex {
    vec4 gl_Position;
};

// vertex shader
void main()
{
	// Note: it is more efficient to calculate mvp_matrix outside
	mat4 modelViewMatrix = ubo.view * mt.model;
	mat4 normalMatrix = transpose(inverse(modelViewMatrix));

	// calculate world(camera) position
	vec4 pos = modelViewMatrix * vec4(inPosition, 1);

	// project to screen
	gl_Position = ubo.proj * pos;
	gl_Position.y = -gl_Position.y;

	// output to fragment shader
	outTexcoord = inTexcoord;
	vec4 n = normalMatrix * vec4(inNormal, 1);
	outNormal = normalize(n.xyz);
	outLightPos = normalize(inverse(ubo.view)* vec4(0.0, 0.0, 1.0, 1.0)).xyz;
	outFragPos = pos.xyz;
	outPosition = inPosition;
}