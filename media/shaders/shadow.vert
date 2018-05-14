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

// Output data ; will be interpolated for each fragment.
layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outViewPos;
layout (location = 2) out vec3 outPos;
layout (location = 3) out vec2 outUV;

out gl_PerVertex {
    vec4 gl_Position;
};

// vertex shader
void main()
{
	outNormal = inNormal;
	outUV = inTexcoord;
	vec3 pos = inPosition;
	outPos = pos;
	outViewPos = (ubo.view * vec4(pos.xyz, 1.0)).xyz;
	gl_Position = ubo.proj * ubo.view * mt.model * vec4(pos.xyz, 1.0);
}